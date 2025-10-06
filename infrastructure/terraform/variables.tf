variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "stock-predictor"
}

variable "container_image" {
  description = "Docker container image"
  type        = string
  default     = "stock-predictor:latest"
}