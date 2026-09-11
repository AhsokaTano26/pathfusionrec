#!/bin/bash
cd /root/pathfusionrec
PYTHONPATH=src:. /root/autodl-tmp/envs/pathfusionrec/bin/python -m unittest discover -s tests -v > /root/autodl-tmp/logs/tiger_ablation_tests.log 2>&1
echo "EXIT=$?" >> /root/autodl-tmp/logs/tiger_ablation_tests.log
