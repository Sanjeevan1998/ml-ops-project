import argparse
import logging
import os
import json
from datetime import datetime
import shutil # For cleaning up downloaded model files
import tempfile # For creating a temporary directory for the model

import torch
from datasets import load_dataset # Kept if user wants to switch data loading
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
import mlflow
import pandas as pd
import boto3 # For S3/Swift object storage interaction
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_from_swift(s3_client, bucket_name, object_key, local_path):
    """Downloads a file from Swift S3 storage."""
    try:
        logger.info(f"Downloading s3://{bucket_name}/{object_key} to {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, object_key, local_path)
        logger.info(f"Successfully downloaded {object_key}")
        return True
    except ClientError as e:
        logger.error(f"Failed to download {object_key} from bucket {bucket_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during download of {object_key}: {e}")
        return False

def setup_local_model_from_swift(base_model_path_s3, local_model_dir_base, s3_endpoint_url, s3_access_key, s3_secret_key):
    """
    Downloads model components from Swift S3 and sets up a local directory.
    Returns the path to the local model directory or None on failure.
    """
    # base_model_path_s3 is like "model/Legal-BERT/"
    # local_model_dir_base is where we'll create the specific temp dir, e.g., /tmp or /work

    # It's good practice to create a unique temporary directory for each run
    local_model_path = tempfile.mkdtemp(dir=local_model_dir_base)
    logger.info(f"Created temporary local model directory: {local_model_path}")

    s3_client = boto3.client(
        's3',
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key
    )

    bucket_name, *model_prefix_parts = base_model_path_s3.split('/', 1)
    if not model_prefix_parts:
        logger.error(f"Invalid S3 model path format: {base_model_path_s3}. Expected bucket/prefix.")
        return None
    model_prefix = model_prefix_parts[0]
    if not model_prefix.endswith('/'): # Ensure prefix ends with a slash
        model_prefix += '/'

    logger.info(f"Target S3 bucket: {bucket_name}, Model prefix: {model_prefix}")

    # Define model file mappings from S3 object store to local structure
    # SentenceTransformer expects specific filenames in the root of the model directory
    files_to_download = {
        f"{model_prefix}legal_bert_base_uncased_statedict.pth": os.path.join(local_model_path, "pytorch_model.bin"), # IMPORTANT: Renaming .pth to pytorch_model.bin
        f"{model_prefix}model_config/config.json": os.path.join(local_model_path, "config.json"),
        f"{model_prefix}tokenizer/special_tokens_map.json": os.path.join(local_model_path, "special_tokens_map.json"),
        f"{model_prefix}tokenizer/tokenizer_config.json": os.path.join(local_model_path, "tokenizer_config.json"),
        f"{model_prefix}tokenizer/tokenizer.json": os.path.join(local_model_path, "tokenizer.json"),
        f"{model_prefix}tokenizer/vocab.txt": os.path.join(local_model_path, "vocab.txt"),
    }

    all_downloads_successful = True
    for s3_key, local_file_path in files_to_download.items():
        if not download_from_swift(s3_client, bucket_name, s3_key, local_file_path):
            all_downloads_successful = False
            break

    if not all_downloads_successful:
        logger.error("One or more model files failed to download. Cleaning up local directory.")
        shutil.rmtree(local_model_path) # Clean up partially downloaded model
        return None

    logger.info(f"All model files successfully downloaded and structured at {local_model_path}")
    return local_model_path

def main(args):
    logger.info(f"Starting fine-tuning with arguments: {args}")

    # Set MLflow tracking URI
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI set to: {args.mlflow_tracking_uri}")

    experiment_name = args.mlflow_experiment_name
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"MLflow experiment '{experiment_name}' created with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
            logger.info(f"MLflow experiment '{experiment_name}' already exists with ID: {experiment_id}. Using this experiment.")
        else:
            raise e
    mlflow.set_experiment(experiment_name)

    local_model_dir_for_sbert = None # To store path of downloaded model
    try: # Use try-finally to ensure cleanup of downloaded model
        with mlflow.start_run(run_name=args.mlflow_run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started with ID: {run_id}")
            mlflow.log_params(vars(args))

            # Download and setup model from Swift S3
            if args.model_name_or_path.startswith("s3://"):
                s3_full_path = args.model_name_or_path[5:] # Remove "s3://"

                # Ensure S3 credentials are provided
                # These should be set as environment variables ideally
                s3_endpoint = args.s3_endpoint_url or os.getenv("AWS_S3_ENDPOINT")
                s3_access_key = args.s3_access_key or os.getenv("AWS_ACCESS_KEY_ID")
                s3_secret_key = args.s3_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")

                if not all([s3_endpoint, s3_access_key, s3_secret_key]):
                    logger.error("S3 endpoint, access key, and secret key must be provided via args or environment variables for S3 model path.")
                    raise ValueError("Missing S3 credentials for model download.")

                # Use /work or /tmp as base for temp model files
                download_base_dir = args.local_model_temp_dir 
                os.makedirs(download_base_dir, exist_ok=True)

                local_model_dir_for_sbert = setup_local_model_from_swift(
                    s3_full_path,
                    download_base_dir,
                    s3_endpoint,
                    s3_access_key,
                    s3_secret_key
                )
                if not local_model_dir_for_sbert:
                    raise RuntimeError("Failed to download and set up model from S3/Swift.")
                model_to_load = local_model_dir_for_sbert
                mlflow.log_param("model_source", args.model_name_or_path) # Log original S3 path
            else: # Assumes model_name_or_path is a HuggingFace ID or a local path already
                model_to_load = args.model_name_or_path
                mlflow.log_param("model_source", "HuggingFace Hub or Pre-downloaded Local")


            logger.info(f"Loading dataset from {args.data_path}")
            try:
                data_df = pd.read_json(args.data_path, lines=True)
                required_cols = ['anchor_text', 'positive_text', 'negative_text']
                if not all(col in data_df.columns for col in required_cols):
                    raise ValueError(f"Data file must contain columns: {required_cols}")
                dataset_list = data_df.to_dict(orient='records')
                logger.info(f"Loaded {len(dataset_list)} records.")
            except Exception as e:
                logger.error(f"Error loading or processing data file: {e}")
                raise

            train_samples = []
            dev_samples = []
            split_idx = int(len(dataset_list) * (1 - args.dev_split_ratio))

            for i, row in enumerate(dataset_list):
                anchor = row['anchor_text']
                positive = row['positive_text']
                negative = row['negative_text']
                if i < split_idx:
                    train_samples.append(InputExample(texts=[anchor, positive, negative]))
                else:
                    dev_samples.append(InputExample(texts=[anchor, positive, negative]))

            logger.info(f"Number of training examples: {len(train_samples)}")
            logger.info(f"Number of development examples: {len(dev_samples)}")
            mlflow.log_param("num_train_samples", len(train_samples))
            mlflow.log_param("num_dev_samples", len(dev_samples))

            logger.info(f"Loading SentenceTransformer model from: {model_to_load}")
            model = SentenceTransformer(model_to_load) # Now loads from local path after download

            train_loss = losses.TripletLoss(model=model)
            train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

            evaluator = None
            if dev_samples:
                evaluator = TripletEvaluator.from_input_examples(dev_samples, name='legal-dev')

            model_output_path_sbert = os.path.join(args.output_dir, f"sbert_model_fit_{run_id}")
            os.makedirs(model_output_path_sbert, exist_ok=True)

            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=args.num_epochs,
                      warmup_steps=args.warmup_steps,
                      output_path=model_output_path_sbert,
                      show_progress_bar=True,
                      evaluation_steps=args.evaluation_steps if dev_samples else 0,
                      checkpoint_path=os.path.join(args.output_dir, f"checkpoints_{run_id}"),
                      checkpoint_save_steps=args.evaluation_steps * 2 if dev_samples else 1000000,
                      checkpoint_save_total_limit=2)

            logger.info("Fine-tuning completed.")
            mlflow.sentence_transformers.log_model(
                sbert_model=model,
                artifact_path="legal-bert-finetuned-triplet"
            )
            logger.info("Trained model logged to MLflow.")

            if evaluator and os.path.exists(os.path.join(model_output_path_sbert, "eval/triplet_evaluation_legal-dev_results.csv")):
                eval_results_df = pd.read_csv(os.path.join(model_output_path_sbert, "eval/triplet_evaluation_legal-dev_results.csv"))
                latest_eval = eval_results_df.iloc[-1]
                # Log main metrics from TripletEvaluator
                mlflow.log_metric("eval_dot_accuracy", latest_eval.get("dot_accuracy", float('nan')))
                mlflow.log_metric("eval_cosine_accuracy", latest_eval.get("cosine_accuracy", float('nan')))
                mlflow.log_metric("eval_manhattan_accuracy", latest_eval.get("manhattan_accuracy", float('nan')))
                mlflow.log_metric("eval_euclidean_accuracy", latest_eval.get("euclidean_accuracy", float('nan')))
                logger.info(f"Evaluation results logged: {latest_eval.to_dict()}")


            logger.info(f"MLflow run {run_id} completed successfully.")

    finally: # Cleanup the downloaded model directory
        if local_model_dir_for_sbert and os.path.exists(local_model_dir_for_sbert):
            logger.info(f"Cleaning up temporary model directory: {local_model_dir_for_sbert}")
            shutil.rmtree(local_model_dir_for_sbert)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Legal-BERT with triplet loss using sentence-transformers, with S3 model download.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL file with triplet data.")
    # Now model_name_or_path can be an S3 path like "s3://bucket/prefix/to/model/" or a HuggingFace ID
    parser.add_argument("--model_name_or_path", type=str, default="nlpaueb/legal-bert-base-uncased", 
                        help="Pre-trained model name (HuggingFace), local path, or S3 path (s3://bucket-name/path/to/model_root/). For S3, ensure the structure matches user's description.")
    parser.add_argument("--local_model_temp_dir", type=str, default="/tmp/downloaded_models", help="Base local directory to temporarily store models downloaded from S3.")

    parser.add_argument("--output_dir", type=str, default="./sbert_output", help="Directory to save sentence-transformers outputs (checkpoints, best model during fit).")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for the learning rate scheduler.")
    parser.add_argument("--dev_split_ratio", type=float, default=0.1, help="Ratio of data to use for development/evaluation (0 to disable).")
    parser.add_argument("--evaluation_steps", type=int, default=500, help="Evaluate every N steps if dev set exists. 0 for no intermediate eval.")

    # MLflow arguments
    parser.add_argument("--mlflow_tracking_uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"), help="MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI env var.")
    parser.add_argument("--mlflow_experiment_name", type=str, default="LegalAI-Triplet-Finetuning", help="Name of the MLflow experiment.")
    parser.add_argument("--mlflow_run_name", type=str, default=f"legalbert-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}", help="Name of the MLflow run.")

    # S3 arguments (can also be set by environment variables)
    parser.add_argument("--s3_endpoint_url", type=str, default=os.getenv("AWS_S3_ENDPOINT"), help="S3 endpoint URL for model download.")
    parser.add_argument("--s3_access_key", type=str, default=os.getenv("AWS_ACCESS_KEY_ID"), help="S3 access key for model download.")
    parser.add_argument("--s3_secret_key", type=str, default=os.getenv("AWS_SECRET_ACCESS_KEY"), help="S3 secret key for model download.")

    args = parser.parse_args()

    if not args.mlflow_tracking_uri:
        try:
            node_ip = os.popen("hostname -I | awk '{print $1}'").read().strip()
            if node_ip:
                args.mlflow_tracking_uri = f"http://{node_ip}:5001"
                logger.warning(f"MLFLOW_TRACKING_URI not set, auto-detected to: {args.mlflow_tracking_uri}. Explicitly set for reliability.")
            else:
                logger.error("Could not auto-detect MLFLOW_TRACKING_URI. Please set it via --mlflow_tracking_uri or environment variable.")
        except Exception as e:
            logger.error(f"Error auto-detecting MLFLOW_TRACKING_URI: {e}. Please set it explicitly.")

    if not args.mlflow_tracking_uri:
        logger.error("MLflow Tracking URI must be provided via --mlflow_tracking_uri argument or MLFLOW_TRACKING_URI environment variable.")
        exit(1)

    main(args)