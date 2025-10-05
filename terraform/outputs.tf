output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_spot_instance_request.nvidia_dev.spot_instance_id
}

output "instance_public_ip" {
  description = "Public IP address of the instance"
  value       = aws_spot_instance_request.nvidia_dev.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the instance"
  value       = aws_spot_instance_request.nvidia_dev.public_dns
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -X -o StrictHostKeyChecking=no ubuntu@${aws_spot_instance_request.nvidia_dev.public_ip}"
}

output "spot_price" {
  description = "Current spot price"
  value       = aws_spot_instance_request.nvidia_dev.spot_price
}

output "ami_id" {
  description = "AMI ID used for the instance"
  value       = var.ami_id
}
