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