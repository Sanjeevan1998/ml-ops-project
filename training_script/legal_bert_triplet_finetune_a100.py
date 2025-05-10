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

        conn_kwargs = {
            'session': sess,
            'os_options': os_options,
            'retries': 5, 
            'starting_backoff': 2, 
        }
        conn = swiftclient.Connection(**conn_kwargs)
        logger.info("Successfully connected to OpenStack Swift.")
        return conn
    except Exception as e:
        logger.error(f"Swift connection failed: {e}")
        return None

def download_from_swift_native(swift_conn, container_name, object_key, local_path):
    """Downloads a single file from Swift."""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        logger.info(f"Attempting to download swift://{container_name}/{object_key} to {local_path}")
        _resp_headers, obj_contents = swift_conn.get_object(container_name, object_key)
        with open(local_path, 'wb') as local_file:
            local_file.write(obj_contents)
        logger.info(f"Successfully downloaded swift://{container_name}/{object_key} to {local_path}")
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
    logger.info(f"Created temporary local model directory for base model: {local_model_path}")

    if not swift_model_prefix.endswith('/'):
        swift_model_prefix += '/'
            
    files_to_download_map = {
        f"{swift_model_prefix}legal_bert_base_uncased_statedict.pth": "pytorch_model.bin",
        f"{swift_model_prefix}model_config/config.json": "config.json",
        f"{swift_model_prefix}tokenizer/special_tokens_map.json": "special_tokens_map.json",
        f"{swift_model_prefix}tokenizer/tokenizer_config.json": "tokenizer_config.json",
        f"{swift_model_prefix}tokenizer/tokenizer.json": "tokenizer.json",
        f"{swift_model_prefix}tokenizer/vocab.txt": "vocab.txt",
    }

    for swift_key, local_filename in files_to_download_map.items():
        if not download_from_swift_native(swift_conn, swift_container_name, swift_key, os.path.join(local_model_path, local_filename)):
            shutil.rmtree(local_model_path) 
            return None
    logger.info(f"All base model files downloaded to {local_model_path}")
    return local_model_path

def upload_directory_to_swift(swift_conn, local_directory, container_name, destination_prefix):
    """Uploads all files from a local directory to a Swift container under a destination prefix."""
    if not swift_conn:
        logger.error("Swift connection not available. Cannot upload model.")
        return False
    
    if not os.path.isdir(local_directory):
        logger.error(f"Local directory {local_directory} not found. Cannot upload.")
        return False

    logger.info(f"Starting upload of directory {local_directory} to swift://{container_name}/{destination_prefix}")
    
    if not destination_prefix.endswith('/'):
        destination_prefix += '/'

    try:

        for root, dirs, files in os.walk(local_directory):
            for filename in files:
                local_filepath = os.path.join(root, filename)
                relative_path = os.path.relpath(local_filepath, local_directory)
                swift_object_name = os.path.join(destination_prefix, relative_path).replace("\\", "/") 

                with open(local_filepath, 'rb') as f:
                    swift_conn.put_object(container_name, swift_object_name, contents=f, headers={'X-Object-Manifest': destination_prefix.strip('/')}) # Added headers for potential large object handling
                logger.info(f"Uploaded {local_filepath} to swift://{container_name}/{swift_object_name}")
        logger.info(f"Successfully uploaded directory {local_directory} to swift://{container_name}/{destination_prefix}")
        return True
    except swiftclient.exceptions.ClientException as e:
        logger.error(f"Swift upload error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Swift upload: {e}")
        return False

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

            if args.model_name_or_path.startswith("swift://") or args.upload_model_to_swift:
                swift_conn = get_swift_connection()
                if not swift_conn:
                    if args.model_name_or_path.startswith("swift://"):
                        raise RuntimeError("Failed to connect to OpenStack Swift for model download.")
                    else:
                        logger.warning("Failed to connect to OpenStack Swift. Model upload will be skipped.")

            if args.model_name_or_path.startswith("swift://"):
                if not swift_conn: 
                     raise RuntimeError("Swift connection required for model download but not established.")
                swift_full_path = args.model_name_or_path[8:]
                try:
                    swift_container_name, swift_model_prefix = swift_full_path.split('/', 1)
                except ValueError:
                    raise ValueError(f"Invalid Swift model path: {args.model_name_or_path}. Expected swift://container/prefix.")

                os.makedirs(args.local_model_temp_dir, exist_ok=True)
                local_model_dir_for_sbert = setup_local_model_from_swift_native(
                    swift_conn, swift_container_name, swift_model_prefix, args.local_model_temp_dir
                )
                if not local_model_dir_for_sbert:
                    raise RuntimeError("Failed to download base model from Swift.")
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
            
            sbert_native_save_path = os.path.join(args.output_dir, f"sbert_finetuned_model_intermediate_{run_id}")
            
            logger.info("Starting model fine-tuning...")
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=args.num_epochs,
                      warmup_steps=args.warmup_steps,
                      output_path=sbert_native_save_path, 
                      show_progress_bar=True,
                      evaluation_steps=args.evaluation_steps if dev_samples and args.evaluation_steps > 0 else 0,
                      checkpoint_path=os.path.join(sbert_native_save_path, "checkpoints"), 
                      checkpoint_save_steps=args.evaluation_steps * 2 if dev_samples and args.evaluation_steps > 0 else 1000000,
                      checkpoint_save_total_limit=2)

            logger.info(f"Fine-tuning completed. Intermediate SBERT saves (best model during fit, checkpoints) are at: {sbert_native_save_path}")
            
            final_model_save_dir = os.path.join(args.output_dir, f"final_sbert_model_{run_id}")
            model.save(final_model_save_dir)
            logger.info(f"Final fine-tuned model explicitly saved locally to: {final_model_save_dir}")
            
            local_save_success_indicator = os.path.join(final_model_save_dir, "_LOCAL_SAVE_SUCCESSFUL.txt")
            with open(local_save_success_indicator, "w") as f:
                f.write(f"Model successfully saved locally to {final_model_save_dir} at {datetime.utcnow().isoformat()}Z")
            try:
                mlflow.log_artifact(local_save_success_indicator, "model_save_status")
                logger.info(f"Logged local save indicator to MLflow artifacts.")
            except Exception as e_mlflow_artifact_log:
                logger.warning(f"Could not log local save indicator to MLflow: {e_mlflow_artifact_log}")


            if args.upload_model_to_swift:
                if swift_conn:
                    if os.path.isdir(final_model_save_dir): 
                        target_swift_prefix = args.swift_upload_prefix or f"models/legal-bert-finetuned/{run_id}"
                        upload_success = upload_directory_to_swift(
                            swift_conn,
                            final_model_save_dir, 
                            args.swift_container_name,
                            target_swift_prefix
                        )
                        if upload_success:
                            swift_uri = f"swift://{args.swift_container_name}/{target_swift_prefix}"
                            mlflow.log_param("finetuned_model_swift_location", swift_uri)
                            logger.info(f"Fine-tuned model uploaded to Swift: {swift_uri} and its location logged to MLflow.")
                        else:
                            logger.error(f"Failed to upload fine-tuned model to Swift. Model is available locally at {final_model_save_dir}")
                            mlflow.log_param("finetuned_model_swift_upload_status", "failed")
                            mlflow.log_param("finetuned_model_local_fallback_path", final_model_save_dir)
                    else:
                        logger.warning(f"Final model save directory {final_model_save_dir} not found. Skipping Swift upload.")
                else: # Swift connection failed earlier
                     logger.warning(f"Swift upload requested but connection is not available. Skipping Swift upload. Model is available locally at {final_model_save_dir}")
                     mlflow.log_param("finetuned_model_swift_upload_status", "skipped_no_connection")
                     mlflow.log_param("finetuned_model_local_fallback_path", final_model_save_dir)
            else: # Swift upload not requested
                logger.info(f"Swift upload not requested. Model is available locally at {final_model_save_dir}")
                mlflow.log_param("finetuned_model_local_path", final_model_save_dir)


            if evaluator:
                eval_csv_path = os.path.join(sbert_native_save_path, "eval/triplet_evaluation_finetuned_legal_dev_results.csv")
                if os.path.exists(eval_csv_path):
                    tuned_eval_df = pd.read_csv(eval_csv_path).iloc[-1]
                    for col, val in tuned_eval_df.items():
                        if isinstance(val, (int, float)): mlflow.log_metric(f"finetuned_model_eval_{col.replace(' ', '_')}", val)
                    try:
                        mlflow.log_artifact(eval_csv_path, "finetuned_model_evaluation_results")
                        logger.info(f"Fine-tuned model evaluation results logged from {eval_csv_path}")
                    except Exception as e_eval_log:
                        logger.warning(f"Could not log evaluation CSV to MLflow: {e_eval_log}")
                else:
                    logger.warning(f"Fine-tuned model evaluation CSV not found at {eval_csv_path}. Check SBERT output path.")
            logger.info(f"MLflow run {run_id} finished.")
    finally: 
        if local_model_dir_for_sbert and os.path.exists(local_model_dir_for_sbert):
            logger.info(f"Cleaning up temporary base model directory: {local_model_dir_for_sbert}")
            shutil.rmtree(local_model_dir_for_sbert)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Legal-BERT with OpenStack Swift and scikit-learn splitting.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL triplet data.")
    parser.add_argument("--model_name_or_path", type=str, default="nlpaueb/legal-bert-base-uncased", 
                        help="Model: HuggingFace ID, local path, or Swift path (swift://container/path/).")
    parser.add_argument("--local_model_temp_dir", type=str, default="/tmp/downloaded_models", help="Base dir for temporary model downloads.")
    parser.add_argument("--output_dir", type=str, default="./sbert_output", help="Base directory for SBERT fit outputs, temp eval files, and final model save for logging.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Learning rate warmup steps.")
    parser.add_argument("--dev_split_ratio", type=float, default=0.1, help="Fraction of data for development set (0 to disable).")
    parser.add_argument("--evaluation_steps", type=int, default=0, help="Evaluate on dev set every N steps (0 to disable intermediate eval).")
    parser.add_argument("--evaluate_base_model", action='store_true', help="Evaluate base model on dev set before fine-tuning.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"), help="MLflow server URI.")
    parser.add_argument("--mlflow_experiment_name", type=str, default="LegalAI-Swift-Sklearn-In-Docker", help="MLflow experiment name.")
    parser.add_argument("--mlflow_run_name", type=str, default=f"sbert-swift-docker-{datetime.now().strftime('%Y%m%d-%H%M%S')}", help="MLflow run name.")
    parser.add_argument("--upload_model_to_swift", action='store_true', help="Upload the fine-tuned model to OpenStack Swift.")
    parser.add_argument("--swift_container_name", type=str, default="object-store-persist-group36", help="Swift container name for uploading the fine-tuned model.")
    parser.add_argument("--swift_upload_prefix", type=str, default=None, help="Prefix (folder path) in Swift container for the fine-tuned model. If None, defaults to 'models/legal-bert-finetuned/RUN_ID'.")

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
