terraform {
  backend "swift" {
    auth_url    = var.auth_url
    user_name   = var.APPLICATION_CREDENTIAL_ID
    password    = vvar.APPLICATION_CREDENTIAL_SECRET
    tenant_name = var.tenant_name
    region_name = var.region_name
    container   = "terraform-tfstate"
    object_name = "terraform.tfstate"
  }
}
