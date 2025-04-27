output "instance_names" {
  value = module.compute.instance_names
}

output "floating_ip" {
  value = module.floatingip.floating_ip_address
}