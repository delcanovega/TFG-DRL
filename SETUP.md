# TFG-DRL
[UCM] TRABAJO DE FIN DE GRADO

## Configuración para desarrollo en local

### Entorno de python

1. Instalar [anaconda 3](https://www.anaconda.com/download).
2. _(Recomendado)_ crear un entorno virtual en el que instalar las librerías necesarias.
``` bash
> conda create --name TFG
```
Recordar activar el entorno antes de instalar librerías en él (se desactiva al apagar/reiniciar o cerrar el terminal).
``` bash
# Desde Windows
> activate TFG  # Si PowerShell da problemas usar cmd o AnacondaPrompt

# Desde Linux / macOS
> source activate TFG

# Si usas fish shell
> conda activate TFG
```
3. Librerías usadas
``` bash
> conda install matplotlib
# Keras debería instalar automáticamente todas las dependencias/librerías: tensorflow, pandas, numpy y scikit-learn
# Si no fuese así es importante que se instalen ANTES de Keras, ya que este reducirá las versiones por compatibilidad
> conda install keras
> conda install pylint
> pip install autopep8  # Si se quiere usar el code formatter
> pip install gym
```

### VS Code

Clonar el repositorio (o abrirlo ya clonado, la pestaña de Source Control debería reconocerlo).

#### Extensiones

* Python (code formatter, linting...)
* LaTeX Workshop (si se va a escribir la memoria desde VS Code)
* Project Manager (para cambiar fácilmente entre el repo de código y el repo de la memoria)

#### Elegir el entorno de python correcto

En la esquina inferior izquierda, hacer click en la versión de Python. Un desplegable se debería abrir desde el que podéis elegir el entorno de Anaconda creado antes.

#### Debuggear

La mejor opción es usar PyCharm.