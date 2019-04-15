ARG USERNAME=vs
FROM $USERNAME/deeplearning:local

ADD . /home/$NB_USER/work/

USER root
RUN chown -R $NB_UID:$NB_GID /home/$NB_USER/work
USER $NB_UID

RUN source ~/.zshrc && python setup.py develop
