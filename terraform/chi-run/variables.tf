variable "suffix" {
  description = "Suffix for resource names (use net ID)"
  type        = string
  nullable = false
  default = "group36"
}

variable "key_pair" {
  description = "Name of key pair"
  type        = string
  default     = "id_rsa_chameleon"
}

variable "reservation_id" {
  description = "Reservation UUID for CHI@UC"
  type        = string
}

variable "node_name" {
  description = "Name of the compute host to schedule this instance on"
  type        = string
}