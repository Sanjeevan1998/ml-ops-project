variable "auth_url" {}
variable "region_name" {}
variable "application_credential_id" {}
variable "application_credential_secret" {}
variable "key_pair" {}
variable "image_name" {}
variable "flavor_name" {}
variable "suffix" {}

variable "nodes" {
  type = map(string)
  default = {
    "node1" = "192.168.1.11"
    "node2" = "192.168.1.12"
    "node3" = "192.168.1.13"
  }
}