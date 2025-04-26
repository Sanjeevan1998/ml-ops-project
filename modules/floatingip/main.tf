resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = "public"
  description = "MLOps IP for deployment"
  port_id     = var.node1_port_id
}