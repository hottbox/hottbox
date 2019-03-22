#!/bin/bash

set -e

### Trigger coverage CI report
if [[ "$COVERAGE" == "true" ]]; then
    coveralls || echo "failed";
fi