resource "openstack_objectstorage_container_v1" "terraform_state" {
  name = "terraform-tfstate-${var.suffix}"
  metadata = {
    "usage" = "terraform-backend"
  }
}
