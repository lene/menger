#!/bin/bash
# List available NVIDIA GPU instance types and their current spot prices

set -e

# Configuration
REGION="${1:-${AWS_REGION:-us-east-1}}"

echo "=== NVIDIA GPU Instance Types in $REGION ==="
echo ""

# NVIDIA instance families
# g4dn: NVIDIA T4 GPUs (cost-effective)
# g5: NVIDIA A10G GPUs (newer, more powerful)
# p3: NVIDIA V100 GPUs (high performance)
# p4d: NVIDIA A100 GPUs (highest performance)
# p5: NVIDIA H100 GPUs (latest generation)

INSTANCE_FAMILIES="g4dn g5 p3 p4d p5"

# Create temporary file for results
TEMP_FILE=$(mktemp)
trap "rm -f $TEMP_FILE" EXIT

echo "Querying instance types and spot prices..."
echo ""

# Query each family
for family in $INSTANCE_FAMILIES; do
  # Get instance types for this family
  aws ec2 describe-instance-types \
    --region "$REGION" \
    --filters "Name=instance-type,Values=${family}.*" \
    --query 'InstanceTypes[].[InstanceType,VCpuInfo.DefaultVCpus,MemoryInfo.SizeInMiB,GpuInfo.Gpus[0].Name,GpuInfo.Gpus[0].Count]' \
    --output json 2>/dev/null | jq -r '.[] | @tsv' | while IFS=$'\t' read -r instance vcpu mem gpu_name gpu_count; do

      # Skip if no results
      [ -z "$instance" ] && continue

      # Get spot price for this instance type
      spot_price=$(aws ec2 describe-spot-price-history \
        --region "$REGION" \
        --instance-types "$instance" \
        --product-descriptions "Linux/UNIX" \
        --max-items 1 \
        --query 'SpotPriceHistory[0].SpotPrice' \
        --output text 2>/dev/null || echo "N/A")

      # Convert memory from MiB to GiB
      mem_gb=$(echo "scale=1; $mem / 1024" | bc)

      # Format and store result
      printf "%-20s %4s vCPUs %6s GiB RAM  %2sx %-20s \$%-8s\n" \
        "$instance" "$vcpu" "$mem_gb" "$gpu_count" "$gpu_name" "$spot_price" >> $TEMP_FILE
    done
done

# Check if we got any results
if [ ! -s "$TEMP_FILE" ]; then
  echo "No NVIDIA GPU instances found in region $REGION"
  echo "This region may not support GPU instances."
  echo ""
  echo "Regions with GPU support: us-east-1, us-west-2, eu-west-1, ap-southeast-1, etc."
  exit 1
fi

# Display results sorted by price
echo "Instance Type         vCPU  RAM         GPU                  Spot Price/hour"
echo "=================================================================================="
sort -t '$' -k2 -n $TEMP_FILE

echo ""
echo "Note: Spot prices are current market rates and change frequently."
echo "      Prices shown are for Linux/UNIX instances."
echo ""
echo "Recommended for development:"
echo "  - g4dn.xlarge:  Most cost-effective, 1x T4 GPU"
echo "  - g4dn.2xlarge: More power, 1x T4 GPU with more CPU/RAM"
echo "  - g5.xlarge:    Newer generation, 1x A10G GPU"
echo ""
