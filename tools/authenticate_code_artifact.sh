#! /bin/sh
#
# Authenticate with AWS codeartifact and generate an access token.
# Pass the access token along to the relevant environment variables
# so that uv can read from our private index.
#
# Usage:
#  . ./tools/authenticate_code_artifact.sh
#
# If the command completes successfully, you will have:
# * a new AWS code artifact access token stored at the
# environment variable UV_INDEX_PRIVATE_REGISTRY_PASSWORD

export AWS_CODEARTIFACT_TOKEN="$(
    aws codeartifact get-authorization-token \
    --domain hum \
    --domain-owner 984625687489 \
    --query authorizationToken \
    --output text
)"

# sigh: https://github.com/astral-sh/uv/issues/9867#issuecomment-2560144648
export UV_INDEX_HUM_CODE_ARTIFACT_USERNAME=aws
export UV_INDEX_HUM_CODE_ARTIFACT_PASSWORD="$AWS_CODEARTIFACT_TOKEN"

# let's make sure pip is authenticated too
# some tools (e.g. pre-commit) still use pip, and our pip config
# at ~/.config/pip/pip.conf points to Code Artifact (not pypi directly)
aws codeartifact login --tool pip --domain hum --repository hum-python
