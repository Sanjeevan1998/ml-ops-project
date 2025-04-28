terraform {
  required_providers {
    openstack = {
      source = "terraform-provider-openstack/openstack"
    }
  }
}

resource "openstack_compute_instance_v2" "nodes" {
  for_each = var.nodes

  name        = "${each.key}-mlops-${var.suffix}"
  image_name  = var.image_name
  flavor_name = var.flavor_name
  key_pair    = var.key_pair

  network {
    port = var.shared_ports[each.key]
  }

  network {
    port = var.private_ports[each.key]
  }

  user_data = <<-EOF
    #!/bin/bash
    echo "127.0.1.1 ${each.key}-mlops-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}
