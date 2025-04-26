terraform {
  required_providers {
    openstack = {
      source = "terraform-provider-openstack/openstack"
    }
  }
}

data "openstack_networking_network_v2" "sharednet1" {
  name = "sharednet1"
}

data "openstack_networking_secgroup_v2" "allow_ssh" {
  name = "default"
}

resource "openstack_networking_port_v2" "private_net_ports" {
  for_each              = var.nodes
  name                  = "port-${each.key}-mlops-${var.suffix}"
  network_id            = var.private_network_id
  port_security_enabled = false

  fixed_ip {
    subnet_id  = var.private_subnet_id
    ip_address = each.value
  }
}

resource "openstack_networking_port_v2" "sharednet1_ports" {
  for_each   = var.nodes
  name       = "sharednet1-${each.key}-mlops-${var.suffix}"
  network_id = data.openstack_networking_network_v2.sharednet1.id
  security_group_ids = [
    data.openstack_networking_secgroup_v2.allow_ssh.id
  ]
}
