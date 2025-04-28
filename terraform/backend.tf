terraform {
  backend "azurerm" {
    resource_group_name   = "primary-resource"
    storage_account_name  = "tfbackendmlopsgroup36"
    container_name        = "terraform-state"
    key                   = "terraform.tfstate" 
  }
}