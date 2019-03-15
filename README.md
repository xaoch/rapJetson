# RAP Video Capture GRPC
Programa para realizar capturas de frames de video y enviar a servidor grpc para su procesamiento

## Requerimientos
- Python v2 o Python v3
- Si se va a utilizar en raspberry con una raspicamera, habilitar la interfaz de la raspicam utilizando '''raspi-config'''

### Requerimientos Python:
Se encuentran en el archivo requirements.txt. Instalar con pip.

# Ejecutar
- Para ejecutar: python main.py
- Para ejecutar como servicio. Crear '''cronjob''' que ejecute archivo cam_capture_service.sh al iniciar la raspberry. Las instrucciones son iguales que en el programa audience_mqtt
- Los topicos mqtt modificarlos en '''topics.py''' para que coincidan con los topicos enviados por '''rap_controller'''
- El servidor y configuraciones de la camara modificar en '''configs.py'''
