variable "region" {
  description = "AWS region to launch the spot instance"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type with NVIDIA GPU"
  type        = string
  default     = "g4dn.xlarge"
}

variable "max_spot_price" {
  description = "Maximum spot price per hour in USD"
  type        = string
  default     = "0.50"
}

variable "max_session_cost" {
  description = "Maximum total session cost in USD (for monitoring/alerting)"
  type        = number
  default     = 10.00
}

variable "ami_id" {
  description = "Custom AMI ID with CUDA 12.8, OptiX, and dev tools (build with scripts/build-ami.sh)"
  type        = string

  validation {
    condition     = can(regex("^ami-[a-f0-9]{8,}$", var.ami_id))
    error_message = "AMI ID must be in format 'ami-xxxxxxxxx'. Build AMI first with scripts/build-ami.sh"
  }
}

variable "user_public_key" {
  description = "SSH public key for access"
  type        = string
  default     = ""
}

variable "auto_terminate" {
  description = "Enable auto-termination on logout"
  type        = bool
  default     = true
}

variable "project_name" {
  description = "Project name for resource tagging"
  type        = string
  default     = "menger-nvidia-dev"
}
