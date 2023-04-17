##
##
ARG UBUNTU_VERSION=20.04
FROM ubuntu:${UBUNTU_VERSION}

#
## set/get locales
RUN set -eux; \
	\
	export DEBIAN_FRONTEND=noninteractive; \
	export DEBCONF_NONINTERACTIVE_SEEN=true; \
	apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

#
# make the "en_US.UTF-8" locale so postgres will be utf-8 enabled by default
ENV LANG en_US.utf8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
#

# We need full control over the running user, including the UID, therefore we
# create the aiuser user id
#
RUN set -eux; \
	\
	export DEBIAN_FRONTEND=noninteractive; \
	export DEBCONF_NONINTERACTIVE_SEEN=true; \
	\
        addgroup --system --gid 4603 aiuser; \
	adduser --uid 4603 -gid 4603 --home /home/aiuser --shell /bin/sh aiuser; \
	\
	apt-get update && apt-get install -y \
	bash \
	curl \
	openssl \
	libssl-dev \
	autoconf \
	automake \
	gnupg \
	gcc \
	g++ \
	pkg-config \
	pkg-config \
	python3 \
	python3-dev \
	python3-pip \
	ssh \
	vim \
	less
#
RUN set -eux; \
	\
	pip3 install --no-cache-dir --upgrade \
	pip \
	setuptools \
	wheel \
	pypi \
	tables \
	tqdm==4.65.0 \
	numpy==1.24.2 \
	pandas==2.0.0 \
	h5py==3.1.0 \
	scikit-learn \
	torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu ;

#
#
# need root to remove the pkgs and clean up
USER root
RUN set -eux; \
	apt-get remove -y --allow-unauthenticated \
		gcc \
		g++ \
		autoconf \
		automake ; \
	apt-get autoremove -y

#
# reset our user to be desired user id
USER aiuser

#
## launch command
CMD ["/bin/bash"]
