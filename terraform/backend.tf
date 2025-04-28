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

terraform {
  backend "swift" {
    auth_url    = var.backend_auth_url          
    user_name   = var.CHITACC_APPLICATION_CREDENTIAL_ID     
    password    = var.CHITACC_APPLICATION_CREDENTIAL_SECRET 
    tenant_name = var.backend_tenant_id          
    region_name = var.backend_region             
    container   = "terraform-tfstate-group36"
    object_name = "terraform.tfstate"
  }
}