variable "auth_url" {}
variable "region_name" {}
variable "domain_name" {}
variable "application_credential_id" {}
variable "application_credential_secret" {}
variable "key_pair" {}
variable "network_name" {}
variable "image_name" {}
variable "flavor_name" {}

# terraform.tfvars.example
# Copy this to terraform.tfvars and fill your secrets
auth_url = "https://chi.uc.chameleoncloud.org:5000/v3"
region_name = "CHI@UC"
domain_name = "Default"
application_credential_id = "your-app-cred-id"
application_credential_secret = "your-app-cred-secret"
key_pair = "your-keypair"
network_name = "sharednet1"
image_name = "CC-Ubuntu20.04"
flavor_name = "m1.medium"