import mlflow
import mlflow.tracking
import pandas as pd 
import os
import argparse
import json
from datetime import datetime
import shutil # For moving files
import logging # For better logging control

try:
    import swiftclient
    from keystoneauth1.identity import v3
    from keystoneauth1 import session as ks_session
except ImportError:
    print("ERROR: Missing OpenStack libraries. Please install: pip install python-swiftclient python-keystoneclient requests")
    exit(1)

# Basic logging configuration
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

def upload_file_to_swift(swift_conn, local_filepath, container_name, swift_object_name):
    """Uploads a single local file to Swift."""
    if not swift_conn:
        logger.error(f"Swift connection not available for uploading {local_filepath}.")
        return False
    if not os.path.exists(local_filepath):
        logger.error(f"Local file {local_filepath} not found for upload.")
        return False
    try:
        logger.info(f"Uploading {local_filepath} to swift://{container_name}/{swift_object_name}...")
        with open(local_filepath, 'rb') as f:
            swift_conn.put_object(container_name, swift_object_name, contents=f)
        logger.info(f"Successfully uploaded {local_filepath} to swift://{container_name}/{swift_object_name}")
        return True
    except swiftclient.exceptions.ClientException as e:
        logger.error(f"Swift upload error for {swift_object_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Swift upload of {swift_object_name}: {e}")
        return False

def export_mlflow_run_data_to_swift(tracking_uri: str, run_id: str, local_temp_output_dir: str, 
                                   upload_to_swift: bool, swift_container: str = None, swift_prefix: str = None):
    """
    Exports MLflow run data locally and optionally uploads to Swift.
    """
    logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
        logger.info(f"Successfully fetched run: {run_id}")
    except Exception as e:
        logger.error(f"Failed to fetch run {run_id}. Error: {e}")
        return

    # Create a temporary local output directory for this run's files
    run_local_dir = os.path.join(local_temp_output_dir, f"mlflow_export_temp_{run_id}")
    os.makedirs(run_local_dir, exist_ok=True)
    logger.info(f"Temporarily saving export data to: {run_local_dir}")

    swift_conn = None
    swift_base_path_for_run = None
    if upload_to_swift:
        if not swift_container or not swift_prefix:
            logger.error("Swift container and prefix must be specified for upload.")
            return # Or proceed with local export only
        swift_conn = get_swift_connection()
        if not swift_conn:
            logger.warning("Failed to connect to Swift. Will only export locally.")
            upload_to_swift = False # Disable upload if connection failed
        else:
            swift_base_path_for_run = os.path.join(swift_prefix, run_id).replace("\\", "/")
            logger.info(f"Target Swift base path for this run: swift://{swift_container}/{swift_base_path_for_run}")


    # 1. Export Parameters
    params_data = run.data.params
    params_file_local = os.path.join(run_local_dir, "parameters.json")
    with open(params_file_local, 'w') as f:
        json.dump(params_data, f, indent=4)
    logger.info(f"Parameters saved locally to: {params_file_local}")
    if upload_to_swift and swift_conn:
        upload_file_to_swift(swift_conn, params_file_local, swift_container, f"{swift_base_path_for_run}/parameters.json")
    
    print(f"\n--- Parameters for Run {run_id} ---")
    for key, value in params_data.items(): print(f"  {key}: {value}")

    # 2. Export Metrics
    full_metrics_history = {}
    for metric_key in run.data.metrics.keys():
        metric_history = client.get_metric_history(run_id, metric_key)
        full_metrics_history[metric_key] = [{"step": m.step, "timestamp": datetime.fromtimestamp(m.timestamp/1000.0).isoformat(), "value": m.value} for m in metric_history]
    
    metrics_file_local = os.path.join(run_local_dir, "metrics_history.json")
    with open(metrics_file_local, 'w') as f:
        json.dump(full_metrics_history, f, indent=4)
    logger.info(f"Metrics (with history) saved locally to: {metrics_file_local}")
    if upload_to_swift and swift_conn:
        upload_file_to_swift(swift_conn, metrics_file_local, swift_container, f"{swift_base_path_for_run}/metrics_history.json")

    print(f"\n--- Final Metrics for Run {run_id} ---")
    for key, value in run.data.metrics.items(): print(f"  {key}: {value}")

    # 3. Export Tags
    tags_data = run.data.tags
    tags_file_local = os.path.join(run_local_dir, "tags.json")
    with open(tags_file_local, 'w') as f:
        json.dump(tags_data, f, indent=4)
    logger.info(f"Tags saved locally to: {tags_file_local}")
    if upload_to_swift and swift_conn:
        upload_file_to_swift(swift_conn, tags_file_local, swift_container, f"{swift_base_path_for_run}/tags.json")

    print(f"\n--- Tags for Run {run_id} ---")
    for key, value in tags_data.items(): print(f"  {key}: {value}")

    # 4. Download and Optionally Upload Specific Known Artifacts
    artifacts_to_process = {
        "base_model_evaluation_results/triplet_evaluation_base_model_dev_eval_results.csv": "base_model_eval.csv",
        "finetuned_model_evaluation_results/triplet_evaluation_finetuned_legal_dev_results.csv": "finetuned_model_eval.csv",
        "model_save_status/_LOCAL_SAVE_SUCCESSFUL.txt": "local_save_indicator.txt"
    }
    
    logger.info("\n--- Processing Artifacts ---")
    for artifact_path_in_mlflow, desired_filename in artifacts_to_process.items():
        local_filepath = os.path.join(run_local_dir, desired_filename) # Save directly with desired name
        try:
            # Ensure local path is clean for download
            if os.path.exists(local_filepath): os.remove(local_filepath)

            client.download_artifacts(
                run_id=run_id,
                path=artifact_path_in_mlflow,
                dst_path=run_local_dir 
            )
            
            downloaded_file_actual_path = os.path.join(run_local_dir, os.path.basename(artifact_path_in_mlflow))
            if os.path.exists(downloaded_file_actual_path):
                if downloaded_file_actual_path != local_filepath:
                    shutil.move(downloaded_file_actual_path, local_filepath)
                
                logger.info(f"Artifact '{artifact_path_in_mlflow}' saved locally as '{local_filepath}'")
                print(f"  Artifact '{artifact_path_in_mlflow}' saved locally as '{local_filepath}'")

                if upload_to_swift and swift_conn:
                    swift_object_name = f"{swift_base_path_for_run}/artifacts/{desired_filename}"
                    upload_file_to_swift(swift_conn, local_filepath, swift_container, swift_object_name)
            else:
                logger.warning(f"Could not find downloaded artifact '{artifact_path_in_mlflow}' at expected temp path.")

        except Exception as e:
            logger.warning(f"Could not download or process artifact '{artifact_path_in_mlflow}'. Error: {e}")
            print(f"  Could not download or process artifact '{artifact_path_in_mlflow}'. Error: {e}")
    
    if upload_to_swift and swift_conn:
        print(f"\nExport to Swift completed for run {run_id}. Check swift://{swift_container}/{swift_base_path_for_run}")
    else:
        print(f"\nLocal export completed for run {run_id}. Files are in: {run_local_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MLflow run data locally and optionally to OpenStack Swift.")
    parser.add_argument("--tracking_uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"), help="MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI env var.")
    parser.add_argument("--run_id", type=str, required=True, help="The ID of the MLflow run to export.")
    parser.add_argument("--local_temp_dir", type=str, default="./mlflow_exports_temp", help="Local temporary directory to save files before potential upload.")
    
    parser.add_argument("--upload_to_swift", action='store_true', help="Enable uploading exported files to Swift.")
    parser.add_argument("--swift_container_name", type=str, default="object-store-persist-group36", help="Swift container name for upload.")
    parser.add_argument("--swift_export_prefix", type=str, default="mlflow_run_exports", help="Base prefix in Swift container for exported run data (run_id will be appended).")
    
    args = parser.parse_args()

    if not args.tracking_uri:
        try:
            logger.warning("MLFLOW_TRACKING_URI not provided via arg or env. Ensure it's correctly set for MLflow client.")
            if not args.tracking_uri:
                 args.tracking_uri = "http://192.5.87.133:8000" # Hardcoding your IP as a last resort for this example
                 logger.warning(f"Using hardcoded MLFLOW_TRACKING_URI: {args.tracking_uri}. Please set it properly.")

        except Exception as e:
            logger.error(f"Could not auto-detect MLFLOW_TRACKING_URI: {e}")
    
    if not args.tracking_uri:
        logger.error("MLflow Tracking URI is essential. Please provide it via --tracking_uri or set MLFLOW_TRACKING_URI environment variable.")
        exit(1)

    if args.upload_to_swift and (not args.swift_container_name or not args.swift_export_prefix):
        logger.error("If --upload_to_swift is set, --swift_container_name and --swift_export_prefix must also be provided.")
        exit(1)

    os.makedirs(args.local_temp_dir, exist_ok=True)

    export_mlflow_run_data_to_swift(
        args.tracking_uri, 
        args.run_id, 
        args.local_temp_dir,
        args.upload_to_swift,
        args.swift_container_name,
        args.swift_export_prefix
    )
