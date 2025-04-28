terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.49.0"
    }
  }
}

provider "openstack" {
  auth_url                     = var.auth_url

  application_credential_id     = var.APPLICATION_CREDENTIAL_ID
  application_credential_secret = var.APPLICATION_CREDENTIAL_SECRET
}
