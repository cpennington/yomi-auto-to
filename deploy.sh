#! /bin/bash

set -e
set -x

ssh pi@raspberrypi.local rm -rf autoto-src
git archive --format=tgz master | ssh pi@raspberrypi.local tar -xz --one-top-level=autoto-src
ssh pi@raspberrypi.local "\
    cd /home/pi/autoto-src && \
    /home/pi/.virtualenvs/autoto/bin/pip-sync && \
    /home/pi/.virtualenvs/autoto/bin/pip install .
    "
