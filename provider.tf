terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"  
      version = "~> 1.49.0"  
    }
  }
}

provider "openstack" {
  auth_url    = var.auth_url
  region      = var.region_name

  application_credential_id     = var.application_credential_id
  application_credential_secret = var.application_credential_secret
}
