
resource "openstack_objectstorage_container_v1" "swift" {
  name = "object-store-persist-${var.suffix}"
}