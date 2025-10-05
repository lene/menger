# Get current VPC (default VPC)
data "aws_vpc" "default" {
  default = true
}

# Security group for SSH and X11 forwarding
resource "aws_security_group" "nvidia_dev" {
  name_prefix = "${var.project_name}-"
  description = "Security group for NVIDIA development spot instance"
  vpc_id      = data.aws_vpc.default.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }

  # X11 forwarding (typically uses SSH tunnel, but included for reference)
  ingress {
    from_port   = 6000
    to_port     = 6063
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "X11 forwarding"
  }

  # Outbound internet access
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  tags = {
    Name    = "${var.project_name}-sg"
    Project = var.project_name
  }
}

# SSH key pair
resource "aws_key_pair" "deployer" {
  key_name_prefix = "${var.project_name}-"
  public_key      = var.user_public_key

  tags = {
    Name    = "${var.project_name}-key"
    Project = var.project_name
  }
}

# Spot instance request
resource "aws_spot_instance_request" "nvidia_dev" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  spot_price             = var.max_spot_price
  wait_for_fulfillment   = true
  spot_type              = "one-time"
  instance_interruption_behavior = "terminate"

  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.nvidia_dev.id]

  user_data = file("${path.module}/user-data.sh")

  root_block_device {
    volume_size           = 100
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name              = "${var.project_name}-spot"
    Project           = var.project_name
    AutoTerminate     = var.auto_terminate
    MaxSessionCost    = var.max_session_cost
  }
}

# Tag the actual instance (spot requests don't propagate tags to instances)
resource "aws_ec2_tag" "instance_tags" {
  for_each = {
    Name           = "${var.project_name}-instance"
    Project        = var.project_name
    AutoTerminate  = var.auto_terminate
    MaxSessionCost = var.max_session_cost
  }

  resource_id = aws_spot_instance_request.nvidia_dev.spot_instance_id
  key         = each.key
  value       = each.value
}
