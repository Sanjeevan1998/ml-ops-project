resource "openstack_objectstorage_container_v1" "terraform_state" {
  name = "terraform-tfstate"
  metadata = {
    "usage" = "terraform-backend"
  }
}
