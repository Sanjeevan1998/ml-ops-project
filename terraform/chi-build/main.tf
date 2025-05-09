
resource "openstack_compute_instance_v2" "reserved_a100" {
  provider    = openstack.uc
  name        = "node-a100-${var.suffix}"
  image_name  = "CC-Ubuntu24.04-CUDA"
  flavor_name = "baremetal"
  key_pair    = var.key_pair
  
  scheduler_hints {
    additional_properties = {
      reservation  = var.reservation_id
      force_hosts  = jsonencode([var.node_name])
    }
  }
}

resource "openstack_networking_floatingip_v2" "fip" {
  provider = openstack.uc
  pool     = "public"
}

resource "openstack_compute_floatingip_associate_v2" "fip_assoc" {
  provider    = openstack.uc
  floating_ip = openstack_networking_floatingip_v2.fip.address
  instance_id = openstack_compute_instance_v2.reserved_a100.id
}