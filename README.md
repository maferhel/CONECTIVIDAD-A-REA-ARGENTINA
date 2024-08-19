![IA_0](https://github.com/maferhel/CONECTIVIDAD-AEREA-ARGENTINA/blob/main/IMAGENES/IA_0.jpg)

<h1 align="center">CONECTIVIDAD AÉREA ARGENTINA</h1>

## CONTEXTO.<br />

Con la sanción del [Decreto de Necesidad y Urgencia 70/2023](https://www.argentina.gob.ar/normativa/nacional/decreto-70-2023-395521/texto), que entró en vigencia a partir del 29 de diciembre de 2023, cuya finalidad esencial es una amplia desregulación del comercio, los servicios y la industria en todo el territorio nacional, derogándose sendas normas que se contrapone a ese principio, entre ellas la [Ley Nro 19030](https://servicios.infoleg.gob.ar/infolegInternet/anexos/130000-134999/130065/texact.htm) de transporte aerocomercial, el mercado de transporte aéreo argentino se ve atravesado por la política denominada "cielos abiertos".<br />

Hasta la sanción de ese DNU, la Ley Nro 19030 disponía que los vuelos de cabotaje (en el territorio nacional) debían ser cubiertos por aerolíneas nacionales y le otorgaba a Aerolíneas Argentinas un mínimo del 50% de esas operaciones e impedía (salvo excepciones) que 2 empresas operen en la misma ruta; tomándosela para los servicios internacionales regulares como instrumento elegido, dotándosela de pautas preferenciales frente a compañías de banderas extranjeras.<br />

Ahora, en cambio se permite que aeronaves extranjeras, con tripulación extranjera, operen en el país, sancionándose en consecuencia el [Decreto 599/2024](https://www.boletinoficial.gob.ar/detalleAviso/primera/310257/20240710), profundizando estos cambios al aprobar un Reglamento de Acceso a los Mercados Aerocomerciales.<br />

En éste contexto, me resultó atractivo analizar los datos del transporte aéreo de pasajeros en el país provenientes del Sistema Integrado de Aviación Civil (SIAC) dependiente de la Administración Nacional de Aviación Civil (ANAC), para efectuar un análisis de lo que ha sido el mercado en los últimos siete años, en vista de una futura medición y comparación de como ésta política aerocomercial impactará en el mercado; datos que se encuentran disponibles en la base de datos pública https://datos.gob.ar/dataset/turismo-conectividad-aerea.<br />

Además, es importante que señale que la información con la que efectué el presente proyecto solo se refiere a información a cerca de vuelos, pasajeros, asientos, rutas y aerolíneas desde el año 2017 al 30 de abril del 2024; limitación que debe contemplarse al momento de leer los resultados graficados en el tablero BI que confeccioné con dicha información, para no tomar esos insights como valores absolutos a cerca de la rentabilidad de las empresas involucradas en el mercado aerocomercial.<br />

## ETAPAS DEL PROYECTO.<br />

**1.- [ANÁLISIS PRELIMINAR](https://github.com/maferhel/CONECTIVIDAD-AEREA-ARGENTINA/blob/main/1.-%20INFORME%20PRELIMINAR.ipynb).**<br />
  
Dentro de ésta etapa, luego de consumir las bases de datos existentes en el repositorio público, se realizó un análisis minucioso del estado de cada dataset, lo que comprendió un informe de cada dataframe, la verificación del tipo de datos existentes en el mismo, un informe de cada columna, visualización de valores nulos, atípicos, outliers y duplicados, para que con dicha información del estado de las bases de datos decidir con cual trabajar en pos de lograr un informe descriptivo del mercado aerocomercial.<br />
  
  
**2.- [TRANSFORMACIÓN DE DATOS](https://github.com/maferhel/CONECTIVIDAD-AEREA-ARGENTINA/blob/main/2.-%20TRANSFORMACI%C3%93N%20DE%20DATOS.ipynb).**<br />
  
A partir de las conclusiones arribadas en el informe preliminar, procedí a realizar las transformaciones que fueron necesarias para optimizar el uso de la información existente en la tabla de datos elegida para la realización del presente proyecto.<br />
  
**3.- [DESARROLLO DE MODELO DE MACHINE LEARNING](https://github.com/maferhel/CONECTIVIDAD-AEREA-ARGENTINA/blob/main/3.-%20MLOPS%20PREDICCIONES.ipynb).**<br />
  
Dado que la información oficial existente data al 30 de abril de 2024, me resultó curioso proyectar como se comportaría el mercado hasta el 31 de diciembre del presente año para realizar una comparación entre la predicciones y el comportamiento real del mercado, que una vez publicados los datos oficiales en el futuro.<br />
  
Por ello en ésta etapa testee tres modelos de entrenamiento de series temporales, para predecir el comportamiento de los vuelos y pasajeros, a saber: **a.-** Regresión Lineal, **b.-** Regresión Polinómica, y **c.-** Modelo fbprophet, optando por utilizar éste último una vez que evalué las métricas de comportamiento de cada uno de ellos, por resultar el ser más certero en sus predicciones.<br />
  
Opté por éste último ya que genera tres columnas clave para cada fecha mensual predicha:<br />
  
- yhat: Es la predicción central o esperada de la cantidad de pasajeros para esa fecha.<br />
- yhat_lower y yhat_upper: Son los límites inferior y superior del intervalo de confianza para la predicción. Estos valores indican el rango dentro del cual se espera que se encuentre la verdadera cantidad de pasajeros, con un cierto nivel de confianza (normalmente el 80% o 95%, dependiendo de la configuración de Prophet).<br />
  
El ancho del intervalo de confianza (yhat_upper - yhat_lower) proporciona una medida de la incertidumbre en la predicción. Un intervalo más estrecho indica mayor confianza en la predicción, mientras que un intervalo más amplio indica mayor incertidumbre.<br />
  
**4.- [VISUALIZACIÓN DE LA INFORMACIÓN EN TABLERO DE POWER BI](https://app.powerbi.com/view?r=eyJrIjoiNDUxNTMwNGYtMTM1Mi00MTRmLWIyYzctYTY4Y2E0MjZkYTgxIiwidCI6ImFlMGI3Mzk4LWFkMmEtNGQ2Ni1iNjQ1LWY4NjlkNTEwYmU2NSIsImMiOjR9&pageName=c4ce6aaea16d40ac6c2a).**<br />
  
En esta etápa dispuse los datos de una forma tal que se respondan los siguientes interrogantes:<br />

- a. Cómo está compuesto el sistema aerocomercial de Argentina?<br />
      
- b. Cómo se comportó el nivel de ocupación a lo largo de los años?
      
- c. Cómo varió el volumen de rutas aéreas y vuelos en el tiempo?
      
- d. Evaluación del mercado de acuerdo a los siguientes indicadores de performance (KPIs):
      
    - *Factor de ocupación:* Esta métrica se calcula dividiendo el número de pasajeros por el número de asientos disponibles en un vuelo, estableciendo un valor razonable entre el 70% y el 80% para alcanzar el punto de equilibrio o break-even.  
        
    $$
    \text{Factor de Ocupación} = \frac{\text{Pasajeros}}{\text{Asientos}}
    $$


    - *Crecimiento de vuelos:* Esto implicó comparar el número de vuelos del trimestre actual, en comparación al trimestre anterior; estableciendo como objetivo un crecimiento trimestral razonable igual al 1%. 


    $$
    \text{Crecimiento de vuelos} = \frac{\text{Número de Vuelos en el Trimestre Actual} - \text{Número de Vuelos en el Trimestre Anterior}}{\text{Número de Vuelos en el Trimestre Anterior}}
    $$


    - *Rentabilidad de la ruta:* Para calcular la rentabilidad de la ruta, se utilizó una fórmula que combinó la frecuencia de la ruta y el porcentaje de ocupación, donde frecuencia de la ruta es el número de vuelos entre un origen y un destino específico en un periodo trimestral.  


    $$
    \text{Rentabilidad} = \text{Frecuencia de la Ruta} \times \text{Factor de Ocupación}
    $$

- e. Cómo se comportará el mercado hasta el 31 de diciembre de 2024 en general y respecto del top ten de aerolíneas que operan en él?

Te invito a explorar el **[dashboard](https://app.powerbi.com/view?r=eyJrIjoiNDUxNTMwNGYtMTM1Mi00MTRmLWIyYzctYTY4Y2E0MjZkYTgxIiwidCI6ImFlMGI3Mzk4LWFkMmEtNGQ2Ni1iNjQ1LWY4NjlkNTEwYmU2NSIsImMiOjR9&pageName=c4ce6aaea16d40ac6c2a)** que preparé para observar el comportamiento de los datos.<br />



## AUTORA.<br />
#### María Fernanda Helguero. <br />
Para cualquier duda/sugerencia/recomendación/mejora respecto al proyecto agradeceré que me la hagas saber, para ello contactame por [LinkedIn](https://www.linkedin.com/in/maria-fernanda-helguero-284087181/)<br />
