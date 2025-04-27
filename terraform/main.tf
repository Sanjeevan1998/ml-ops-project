module "network" {
  source = "./modules/network"
  suffix = var.suffix
}

module "ports" {
  source = "./modules/ports"
  nodes = var.nodes
  suffix = var.suffix
  private_network_id = module.network.private_net_id
  private_subnet_id = module.network.private_subnet_id
}

module "compute" {
  source = "./modules/compute"
  nodes = var.nodes
  suffix = var.suffix
  image_name = var.image_name
  flavor_name = var.flavor_name
  key_pair = var.key_pair
  private_ports = module.ports.private_ports
  shared_ports = module.ports.shared_ports
}

module "floatingip" {
  source = "./modules/floatingip"
  node1_port_id = module.ports.shared_ports["node1"]
}
