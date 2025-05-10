
import argparse
import logging
import os
import json
from datetime import datetime
import shutil
import tempfile
import random 
import numpy as np

import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import swiftclient
    from keystoneauth1.identity import v3
    from keystoneauth1 import session as ks_session
except ImportError:
    print("ERROR: Missing OpenStack libraries. Please install: pip install python-swiftclient python-keystoneclient requests scikit-learn")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_swift_connection():
    """Establishes a connection to OpenStack Swift using environment variables."""
    try:
        auth_url = os.environ.get('OS_AUTH_URL')
        app_cred_id = os.environ.get('OS_APPLICATION_CREDENTIAL_ID')
        app_cred_secret = os.environ.get('OS_APPLICATION_CREDENTIAL_SECRET')
        project_id = os.environ.get('OS_PROJECT_ID')
        project_name = os.environ.get('OS_PROJECT_NAME')
        project_domain_name = os.environ.get('OS_PROJECT_DOMAIN_NAME', 'Default')

        if not all([auth_url, app_cred_id, app_cred_secret]):
            logger.error("Missing OpenStack auth env vars (OS_AUTH_URL, OS_APPLICATION_CREDENTIAL_ID, OS_APPLICATION_CREDENTIAL_SECRET). Source your RC file.")
            return None

        auth_args = {
            'auth_url': auth_url,
            'application_credential_id': app_cred_id,
            'application_credential_secret': app_cred_secret,
        }
        if project_id:
            auth_args['project_id'] = project_id
        elif project_name and project_domain_name:
             auth_args['project_name'] = project_name
             auth_args['project_domain_name'] = project_domain_name
        
        auth = v3.ApplicationCredential(**auth_args)
        sess = ks_session.Session(auth=auth, timeout=60)
        
        os_options = {
            "project_id": project_id,
            "user_domain_name": os.environ.get('OS_USER_DOMAIN_NAME', 'Default'),
            "project_domain_name": project_domain_name,
            "region_name": os.environ.get('OS_REGION_NAME'),
            "interface": os.environ.get('OS_INTERFACE', 'public'),
            "identity_api_version": "3",
        }
        os_options = {k: v for k, v in os_options.items() if v is not None}

        conn = swiftclient.Connection(session=sess, os_options=os_options)
        logger.info("Successfully connected to OpenStack Swift.")
        return conn
    except Exception as e:
        logger.error(f"Swift connection failed: {e}")
        return None

def download_from_swift_native(swift_conn, container_name, object_key, local_path):
    """Downloads a single file from Swift."""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        _resp_headers, obj_contents = swift_conn.get_object(container_name, object_key)
        with open(local_path, 'wb') as local_file:
            local_file.write(obj_contents)
        logger.info(f"Downloaded swift://{container_name}/{object_key} to {local_path}")
        return True
    except swiftclient.exceptions.ClientException as e:
        logger.error(f"Swift download error for {object_key}: {e} (Status: {e.http_status if hasattr(e, 'http_status') else 'N/A'})")
        return False
    except Exception as e:
        logger.error(f"Unexpected download error for {object_key}: {e}")
        return False

def setup_local_model_from_swift_native(swift_conn, swift_container_name, swift_model_prefix, local_model_dir_base):
    """Downloads model components from Swift to a temporary local directory."""
    local_model_path = tempfile.mkdtemp(dir=local_model_dir_base)
    logger.info(f"Created temporary local model directory: {local_model_path}")

    if not swift_model_prefix.endswith('/'):
        swift_model_prefix += '/'
            
    files_to_download_map = {
        f"{swift_model_prefix}legal_bert_base_uncased_statedict.pth": "pytorch_model.bin", # Renamed for SBERT
        f"{swift_model_prefix}model_config/config.json": "config.json",
        f"{swift_model_prefix}tokenizer/special_tokens_map.json": "special_tokens_map.json",
        f"{swift_model_prefix}tokenizer/tokenizer_config.json": "tokenizer_config.json",
        f"{swift_model_prefix}tokenizer/tokenizer.json": "tokenizer.json",
        f"{swift_model_prefix}tokenizer/vocab.txt": "vocab.txt",
    }

    for swift_key, local_filename in files_to_download_map.items():
        if not download_from_swift_native(swift_conn, swift_container_name, swift_key, os.path.join(local_model_path, local_filename)):
            shutil.rmtree(local_model_path) # Cleanup on failure
            return None
    logger.info(f"All model files downloaded to {local_model_path}")
    return local_model_path


def main(args):
    logger.info(f"Fine-tuning run started with args: {args}")

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    logger.info(f"Random seed set to {args.random_seed}")

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    logger.info(f"Using MLflow experiment '{args.mlflow_experiment_name}'")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Ensured output directory exists: {args.output_dir}")

    local_model_dir_for_sbert = None 
    swift_conn = None

    try: 
        with mlflow.start_run(run_name=args.mlflow_run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            mlflow.log_params({k: str(v) if v is not None else "" for k, v in vars(args).items()})

            if args.model_name_or_path.startswith("swift://"):
                swift_full_path = args.model_name_or_path[8:]
                try:
                    swift_container_name, swift_model_prefix = swift_full_path.split('/', 1)
                except ValueError:
                    raise ValueError(f"Invalid Swift model path: {args.model_name_or_path}. Expected swift://container/prefix.")

                swift_conn = get_swift_connection()
                if not swift_conn:
                    raise RuntimeError("Failed to connect to OpenStack Swift for model download.")

                os.makedirs(args.local_model_temp_dir, exist_ok=True)
                local_model_dir_for_sbert = setup_local_model_from_swift_native(
                    swift_conn, swift_container_name, swift_model_prefix, args.local_model_temp_dir
                )
                if not local_model_dir_for_sbert:
                    raise RuntimeError("Failed to download model from Swift.")
                model_to_load = local_model_dir_for_sbert
                mlflow.log_param("model_source_swift_path", args.model_name_or_path)
            else: 
                model_to_load = args.model_name_or_path
                mlflow.log_param("model_source_local_or_hf", args.model_name_or_path)
            
            logger.info(f"Loading dataset from {args.data_path}")
            try:
                dataset_list = pd.read_json(args.data_path, lines=True).to_dict(orient='records')
                if not dataset_list or not all(k in dataset_list[0] for k in ['anchor_text', 'positive_text', 'negative_text']):
                    raise ValueError("Dataset is empty or missing required triplet keys.")
            except Exception as e:
                raise RuntimeError(f"Error loading or parsing data file {args.data_path}: {e}")

            if args.dev_split_ratio > 0:
                train_data, dev_data = train_test_split(
                    dataset_list, 
                    test_size=args.dev_split_ratio, 
                    random_state=args.random_seed, 
                    shuffle=True
                )
            else:
                train_data = dataset_list
                dev_data = []
            
            train_samples = [InputExample(texts=[row['anchor_text'], row['positive_text'], row['negative_text']]) for row in train_data]
            dev_samples = [InputExample(texts=[row['anchor_text'], row['positive_text'], row['negative_text']]) for row in dev_data]
            
            logger.info(f"Train samples: {len(train_samples)}, Dev samples: {len(dev_samples)}")
            mlflow.log_param("num_train_samples", len(train_samples))
            mlflow.log_param("num_dev_samples", len(dev_samples))

            logger.info(f"Loading SentenceTransformer model from: {model_to_load}")
            model = SentenceTransformer(model_to_load)

            if dev_samples and args.evaluate_base_model:
                logger.info("Evaluating base model on development set...")
                base_model_evaluator = TripletEvaluator.from_input_examples(dev_samples, name='base_model_dev_eval')
                os.makedirs(args.output_dir, exist_ok=True)
                temp_base_eval_path = tempfile.mkdtemp(dir=args.output_dir)
                logger.info(f"Created temporary directory for base model evaluation: {temp_base_eval_path}")

                base_model_evaluator(model, output_path=temp_base_eval_path)
                
                base_eval_csv_path = os.path.join(temp_base_eval_path, "triplet_evaluation_base_model_dev_eval_results.csv")
                if os.path.exists(base_eval_csv_path):
                    base_eval_df = pd.read_csv(base_eval_csv_path).iloc[-1]
                    for col, val in base_eval_df.items():
                        if isinstance(val, (int, float)): mlflow.log_metric(f"base_model_eval_{col.replace(' ', '_')}", val)
                    mlflow.log_artifact(base_eval_csv_path, "base_model_evaluation_results")
                    logger.info(f"Base model evaluation results logged from {base_eval_csv_path}")
                else:
                    logger.warning(f"Base model evaluation CSV not found at {base_eval_csv_path}")
                shutil.rmtree(temp_base_eval_path)
                logger.info(f"Cleaned up temporary base model evaluation directory: {temp_base_eval_path}")


            train_loss = losses.TripletLoss(model=model)
            train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
            
            evaluator = None
            if dev_samples:
                evaluator = TripletEvaluator.from_input_examples(dev_samples, name='finetuned_legal_dev')
            
            model_output_path_sbert = os.path.join(args.output_dir, f"sbert_model_fit_{run_id}")
            
            logger.info("Starting model fine-tuning...")
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=args.num_epochs,
                      warmup_steps=args.warmup_steps,
                      output_path=model_output_path_sbert,
                      show_progress_bar=True,
                      evaluation_steps=args.evaluation_steps if dev_samples and args.evaluation_steps > 0 else 0,
                      checkpoint_path=os.path.join(args.output_dir, f"checkpoints_{run_id}"),
                      checkpoint_save_steps=args.evaluation_steps * 2 if dev_samples and args.evaluation_steps > 0 else 1000000,
                      checkpoint_save_total_limit=2)

            logger.info("Fine-tuning completed.")
            mlflow.sentence_transformers.log_model(sbert_model=model, artifact_path="legal-bert-finetuned-triplet")

            if evaluator:
                eval_csv_path = os.path.join(model_output_path_sbert, "eval/triplet_evaluation_finetuned_legal_dev_results.csv")
                if os.path.exists(eval_csv_path):
                    tuned_eval_df = pd.read_csv(eval_csv_path).iloc[-1]
                    for col, val in tuned_eval_df.items():
                        if isinstance(val, (int, float)): mlflow.log_metric(f"finetuned_model_eval_{col.replace(' ', '_')}", val)
                    mlflow.log_artifact(eval_csv_path, "finetuned_model_evaluation_results")
                    logger.info(f"Fine-tuned model evaluation results logged from {eval_csv_path}")
                else:
                    logger.warning(f"Fine-tuned model evaluation CSV not found at {eval_csv_path}")
            logger.info(f"MLflow run {run_id} finished.")
    finally: 
        if local_model_dir_for_sbert and os.path.exists(local_model_dir_for_sbert):
            logger.info(f"Cleaning up temporary model directory: {local_model_dir_for_sbert}")
            shutil.rmtree(local_model_dir_for_sbert)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Legal-BERT with OpenStack Swift and scikit-learn splitting.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL triplet data.")
    parser.add_argument("--model_name_or_path", type=str, default="nlpaueb/legal-bert-base-uncased", 
                        help="Model: HuggingFace ID, local path, or Swift path (swift://container/path/).")
    parser.add_argument("--local_model_temp_dir", type=str, default="/tmp/downloaded_models", help="Base dir for temporary model downloads.")
    parser.add_argument("--output_dir", type=str, default="./sbert_output", help="Directory for SBERT fit outputs and temp eval files.") # Clarified help
    parser.add_argument("--num_epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Learning rate warmup steps.")
    parser.add_argument("--dev_split_ratio", type=float, default=0.1, help="Fraction of data for development set (0 to disable).")
    parser.add_argument("--evaluation_steps", type=int, default=0, help="Evaluate on dev set every N steps (0 to disable intermediate eval).")
    parser.add_argument("--evaluate_base_model", action='store_true', help="Evaluate base model on dev set before fine-tuning.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"), help="MLflow server URI.")
    parser.add_argument("--mlflow_experiment_name", type=str, default="LegalAI-Swift-Sklearn-In-Docker", help="MLflow experiment name.") # Updated default
    parser.add_argument("--mlflow_run_name", type=str, default=f"sbert-swift-docker-{datetime.now().strftime('%Y%m%d-%H%M%S')}", help="MLflow run name.") # Updated default
    
    args = parser.parse_args()

    if not (0.0 <= args.dev_split_ratio < 1.0):
        raise ValueError("dev_split_ratio must be between 0.0 and <1.0")
    if not args.mlflow_tracking_uri: 
        try:
            node_ip = os.popen("hostname -I | awk '{print $1}'").read().strip()
            if node_ip: args.mlflow_tracking_uri = f"http://{node_ip}:8000"
        except Exception: pass 
        if not args.mlflow_tracking_uri:
            logger.error("MLflow Tracking URI must be provided via --mlflow_tracking_uri or auto-detection.")
            exit(1)
            
    main(args)
