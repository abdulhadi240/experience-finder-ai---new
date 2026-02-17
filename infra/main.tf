########################################
# Terraform + AWS Provider
########################################
terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
  	   source  = "hashicorp/random"
  	   version = "~> 3.6"
	}
  }

  backend "s3" {
    bucket         = "agentic-terraform-state-ai"  # S3 bucket for state
    key            = "ecs/terraform.tfstate"       # Path inside the bucket
    region         = "us-east-1"
    dynamodb_table = "agentic-terraform-locks"     # State lock table
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}

########################################
# Variables
########################################
variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "agentic-api"
}

variable "container_port" {
  type    = number
  default = 8080
}

variable "desired_count" {
  type    = number
  default = 2
}

variable "min_capacity" {
  type    = number
  default = 2
}

variable "max_capacity" {
  type    = number
  default = 6
}

variable "cpu_target" {
  type    = number
  default = 60
}

variable "memory" {
  type    = number
  default = 1024
}

variable "cpu" {
  type    = number
  default = 512
}

variable "container_image" {
  type    = string
  default = ""
}

# SSM parameter names for secrets
variable "ssm_param_openai" {
  type    = string
  default = "/agentic/OPENAI_API_KEY"
}

variable "ssm_param_zep" {
  type    = string
  default = "/agentic/ZEP_API_KEY"
}

# variable "ssm_param_tavily" {
#   type    = string
#   default = "/agentic/TAVILY_API_KEY"
# }

variable "acm_certificate_arn" {
  type        = string
  description = "ARN of the ACM certificate for hiptraveler.com"
}

variable "redis_node_type" {
  type    = string
  default = "cache.t4g.medium"
}

variable "redis_engine_version" {
  type    = string
  default = "7.1"
}

variable "ssm_param_redis_auth_token" {
  type    = string
  default = "/agentic/REDIS_AUTH_TOKEN"
}

########################################
# AWS SSM Parameters (for ECS Secrets)
########################################

variable "ssm_param_google_maps" {
  type    = string
  default = "/agentic/GOOGLE_MAPS_API_KEY"
}

variable "ssm_param_supabase_url" {
  type    = string
  default = "/agentic/SUPABASE_URL"
}

variable "ssm_param_supabase_project_id" {
  type    = string
  default = "/agentic/SUPABASE_PROJECT_ID"
}

variable "ssm_param_supabase_key" {
  type    = string
  default = "/agentic/SUPABASE_KEY"
}

variable "ssm_param_supabase_service_role_key" {
  type    = string
  default = "/agentic/SUPABASE_SERVICE_ROLE_KEY"
}

variable "ssm_param_perplexity_api_key" {
  type    = string
  default = "/agentic/PERPLEXITY_API_KEY"
}

########################################
# Networking (VPC, Subnets, Routing)
########################################
data "aws_availability_zones" "available" {}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "${var.project_name}-vpc" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.0.0/20"
  map_public_ip_on_launch = true
  availability_zone       = data.aws_availability_zones.available.names[0]
  tags = { Name = "${var.project_name}-public-a" }
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.16.0/20"
  map_public_ip_on_launch = true
  availability_zone       = data.aws_availability_zones.available.names[1]
  tags = { Name = "${var.project_name}-public-b" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${var.project_name}-public-rt" }
}

resource "aws_route_table_association" "a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

########################################
# Security Groups
########################################
resource "aws_security_group" "alb_sg" {
  name        = "${var.project_name}-alb-sg"
  description = "Security group for Application Load Balancer"
  vpc_id      = aws_vpc.main.id

  # Allow HTTP (port 80)
  ingress {
    description = "Allow HTTP from anywhere"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow HTTPS (port 443)
  ingress {
    description = "Allow HTTPS from anywhere"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-alb-sg" }
}

resource "aws_security_group" "service_sg" {
  name        = "${var.project_name}-svc-sg"
  description = "Security group for ECS Fargate service"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Allow traffic from ALB"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-service-sg" }
}

########################################
# Redis (ElastiCache)
########################################

resource "aws_security_group" "redis_sg" {
  name        = "${var.project_name}-redis-sg"
  description = "Security group for Redis"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Redis from ECS service"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.service_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-redis-sg" }
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.project_name}-redis-subnets"
  subnet_ids = [aws_subnet.public_a.id, aws_subnet.public_b.id]
}

resource "random_password" "redis_auth_token" {
  length  = 32
  special = false
}

resource "aws_ssm_parameter" "redis_auth_token" {
  name        = var.ssm_param_redis_auth_token
  description = "Redis AUTH token for ${var.project_name}"
  type        = "SecureString"
  value       = random_password.redis_auth_token.result
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = replace("${var.project_name}-redis", "_", "-")
  description                = "Redis for ${var.project_name} sessions"
  engine                     = "redis"
  engine_version             = var.redis_engine_version
  node_type                  = var.redis_node_type
  port                       = 6379

  num_cache_clusters         = 1
  automatic_failover_enabled = false

  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  security_group_ids         = [aws_security_group.redis_sg.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = aws_ssm_parameter.redis_auth_token.value
}

output "redis_primary_endpoint" {
  value = aws_elasticache_replication_group.redis.primary_endpoint_address
}


########################################
# Load Balancer + Target Group + Listener
########################################
resource "aws_lb" "app" {
  name               = "${var.project_name}-alb"
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]
}

resource "aws_lb_target_group" "app_tg" {
  name        = "${var.project_name}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id

  health_check {
    path              = "/health"
    protocol          = "HTTP"
    matcher           = "200-399"
    interval          = 15
    healthy_threshold = 2
    unhealthy_threshold = 3
    timeout           = 5
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

########################################
# HTTPS Listener (Port 443)
########################################
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.app.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app_tg.arn
  }
}


########################################
# ECR Repository (read existing)
########################################
# data "aws_ecr_repository" "repo" {
#   name = var.project_name
# }

resource "aws_ecr_repository" "repo" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "${var.project_name}-repo"
  }
}

########################################
# ECS Cluster + IAM Roles
########################################
resource "aws_ecs_cluster" "this" {
  name = "${var.project_name}-cluster"
}

# Task execution role (for pulling image & reading secrets)
resource "aws_iam_role" "task_exec_role" {
  name = "${var.project_name}-task-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "ecs-tasks.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "exec_logs" {
  role       = aws_iam_role.task_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_policy" "ssm_read" {
  name = "${var.project_name}-ssm-read"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect   = "Allow",
      Action   = ["ssm:GetParameters", "ssm:GetParameter", "kms:Decrypt"],
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "exec_ssm" {
  role       = aws_iam_role.task_exec_role.name
  policy_arn = aws_iam_policy.ssm_read.arn
}

########################################
# CloudWatch Logs
########################################
resource "aws_cloudwatch_log_group" "logs" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 14
}

########################################
# ECS Task Definition + Service removed ignore image
########################################
locals {
  container_name = var.project_name
}

resource "aws_ecs_task_definition" "task" {
  family                   = "${var.project_name}-td"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.cpu)
  memory                   = tostring(var.memory)
  execution_role_arn       = aws_iam_role.task_exec_role.arn
  task_role_arn            = aws_iam_role.task_exec_role.arn

  container_definitions = jsonencode([
    {
      name      = local.container_name,
      image     = var.container_image,
      essential = true,
      portMappings = [{
        containerPort = var.container_port,
        hostPort      = var.container_port,
        protocol      = "tcp"
      }],
      environment = [
        { name = "APP_ENV", value = "prod" },
        { name = "REDIS_HOST", value = aws_elasticache_replication_group.redis.primary_endpoint_address },
		{ name = "REDIS_PORT", value = "6379" },
		{ name = "REDIS_USE_TLS", value = "true" },
		{ name = "REDIS_ENABLED", value = "false" },

		# Your app knobs:
		{ name = "REDIS_TTL_SECONDS", value = "3600" },
		{ name = "REDIS_IDLE_CUTOFF_SECONDS", value = "1200" },
		{ name = "REDIS_OLD_INTERACTIONS_LIMIT", value = "3" },
		{ name = "REDIS_OLD_INTERACTIONS_MAX", value = "30" },
      ],
      secrets = [
        { name = "OPENAI_API_KEY", valueFrom = var.ssm_param_openai },
        { name = "ZEP_API_KEY",    valueFrom = var.ssm_param_zep },
        { name = "GOOGLE_MAPS_API_KEY",        valueFrom = var.ssm_param_google_maps },
        { name = "SUPABASE_URL",               valueFrom = var.ssm_param_supabase_url },
        { name = "SUPABASE_PROJECT_ID",        valueFrom = var.ssm_param_supabase_project_id },
        { name = "SUPABASE_KEY",               valueFrom = var.ssm_param_supabase_key },
        { name = "SUPABASE_SERVICE_ROLE_KEY",  valueFrom = var.ssm_param_supabase_service_role_key },
        { name = "PERPLEXITY_API_KEY",         valueFrom = var.ssm_param_perplexity_api_key },
        { name = "REDIS_AUTH_TOKEN", valueFrom = var.ssm_param_redis_auth_token },
      ],
      logConfiguration = {
        logDriver = "awslogs",
        options = {
          awslogs-group         = aws_cloudwatch_log_group.logs.name,
          awslogs-region        = var.aws_region,
          awslogs-stream-prefix = "ecs"
        }
      },
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.container_port}/health || exit 1"],
        interval    = 15,
        timeout     = 5,
        retries     = 3,
        startPeriod = 10
      }
    }
  ])
}

resource "aws_ecs_service" "svc" {
  name            = "${var.project_name}-svc"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.task.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  platform_version = "1.4.0"
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  health_check_grace_period_seconds  = 60

  network_configuration {
    subnets         = [aws_subnet.public_a.id, aws_subnet.public_b.id]
    security_groups = [aws_security_group.service_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app_tg.arn
    container_name   = local.container_name
    container_port   = var.container_port
  }

  depends_on = [
    aws_lb_listener.http,
    aws_ecs_task_definition.task
  ]
}


########################################
# Autoscaling (CPU-based)
########################################
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = var.max_capacity
  min_capacity       = var.min_capacity
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.svc.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.project_name}-cpu-policy"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = var.cpu_target
    scale_in_cooldown  = 60
    scale_out_cooldown = 60
  }
}

########################################
# Outputs
########################################
output "alb_dns_name" {
  description = "Public DNS name of the Application Load Balancer"
  value       = aws_lb.app.dns_name
}

output "ecr_repo_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.repo.repository_url
}