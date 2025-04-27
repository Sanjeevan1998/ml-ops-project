variable "nodes" { type = map(string) }
variable "suffix" {}
variable "image_name" {}
variable "flavor_name" {}
variable "key_pair" {}
variable "private_ports" { type = map(string) }
variable "shared_ports" { type = map(string) }