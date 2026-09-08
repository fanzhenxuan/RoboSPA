#!/bin/bash

object_name=${1}
object_id=${2}

if [ -z "$object_name" ]; then
    echo "Error: object_name is required."
    echo "Usage: $0 <object_name> [object_id]"
    exit 1
fi

if [ -z "$object_id" ]; then
    python utils/generate_object_description.py "$object_name" 
else
    python utils/generate_object_description.py "$object_name" --index "$object_id"
fi