#!/bin/bash

set -e

if [[ "$COVERAGE" == "true" ]]; then
    coveralls || echo "failed";
fi