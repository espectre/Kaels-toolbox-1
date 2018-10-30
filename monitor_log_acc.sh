#!/bin/bash
watch -n 30 'cat '${1}'|grep -E "Validation|Current"|tail -n 34'
