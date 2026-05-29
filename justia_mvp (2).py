"""
================================================================================
JustIA - MVP: Clasificación Automática de Textos Legales
================================================================================
Proyecto de Aplicación - Asturias Corporación Universitaria
Asignatura: Inteligencia Artificial para el Desarrollo de Software

Descripción:
    Módulo funcional que clasifica automáticamente consultas y textos jurídicos
    en categorías temáticas (Familia, Laboral, Penal, Civil, Administrativo)
    usando TF-IDF + Naive Bayes, con datos sintéticos representativos del
    contexto colombiano.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DATOS JURÍDICOS SINTÉTICOS (contexto colombiano) — 150 muestras
# ==============================================================================

datos_juridicos = [
    # ---- DERECHO DE FAMILIA (30 muestras) ----
    # Originales
    ("Solicito custodia compartida de mis hijos menores tras el divorcio.", "Familia"),
    ("Mi esposa pide alimentos para los niños, ¿cuánto debo pagar?", "Familia"),
    ("Quiero iniciar proceso de divorcio por mutuo acuerdo.", "Familia"),
    ("Necesito reconocer la paternidad de mi hijo mediante prueba de ADN.", "Familia"),
    ("Mi ex pareja no me permite ver a mis hijos en las fechas acordadas.", "Familia"),
    ("Solicito medida de protección por violencia intrafamiliar.", "Familia"),
    ("Deseo adoptar un menor que está en situación de abandono.", "Familia"),
    ("Mi cónyuge oculta bienes durante el proceso de liquidación de sociedad conyugal.", "Familia"),
    ("Necesito asesoría para capitulaciones matrimoniales.", "Familia"),
    ("Reclamo alimentos para mi madre adulta mayor que no puede trabajar.", "Familia"),
    # Nuevas
    ("Quiero demandar a mi esposo por incumplimiento del régimen de visitas.", "Familia"),
    ("Mi pareja y yo nos separamos y necesitamos dividir los bienes gananciales.", "Familia"),
    ("Solicito aumento de la cuota alimentaria para mis hijos porque los gastos subieron.", "Familia"),
    ("Necesito tramitar la tutela de mi sobrino huérfano para cuidarlo legalmente.", "Familia"),
    ("Mi exesposo no paga la pensión alimentaria hace cuatro meses.", "Familia"),
    ("Quiero impedir que mi ex se lleve a los niños al exterior sin mi autorización.", "Familia"),
    ("Deseo hacer la unión marital de hecho con mi compañero permanente.", "Familia"),
    ("Mi padre falleció sin testamento y necesito iniciar sucesión intestada.", "Familia"),
    ("Quiero demandar a mi hermano por no reconocer mis derechos hereditarios.", "Familia"),
    ("Necesito orientación para la interdicción judicial de mi familiar con discapacidad mental.", "Familia"),
    ("Me acusan de abandono de hogar y quiero defenderme en el proceso de divorcio.", "Familia"),
    ("Quiero solicitar medida de aseguramiento para proteger a mis hijos del padre abusivo.", "Familia"),
    ("Necesito cambiar el apellido de mi hijo porque el padre no lo reconoció.", "Familia"),
    ("Mi esposa solicitó divorcio contencioso alegando incumplimiento de deberes conyugales.", "Familia"),
    ("Quiero impugnar la paternidad del hijo que registré pero no es mío.", "Familia"),
    ("Solicito reducción de cuota alimentaria porque perdí mi trabajo.", "Familia"),
    ("Necesito asesoría para tramitar la adopción de un menor extranjero.", "Familia"),
    ("Mi expareja me impide ver a nuestra hija a pesar de la sentencia judicial.", "Familia"),
    ("Quiero disolver la sociedad patrimonial que tengo con mi compañero permanente.", "Familia"),
    ("Me separé de mi esposo y necesito definir quién se queda con la casa familiar.", "Familia"),

    # ---- DERECHO LABORAL (30 muestras) ----
    # Originales
    ("Me despidieron sin justa causa después de 5 años en la empresa.", "Laboral"),
    ("No me han pagado las horas extras del último mes.", "Laboral"),
    ("Mi empleador no me quiere afiliar a salud y pensión.", "Laboral"),
    ("Sufrí un accidente de trabajo y la ARL no reconoce la incapacidad.", "Laboral"),
    ("Me acosan laboralmente y el jefe me presiona para renunciar.", "Laboral"),
    ("No me pagaron las prestaciones sociales al terminar el contrato.", "Laboral"),
    ("Quiero demandar a mi empresa por despido discriminatorio por embarazo.", "Laboral"),
    ("Mi contrato a término fijo no fue renovado y tengo fuero sindical.", "Laboral"),
    ("El empleador me descontó el salario sin autorización.", "Laboral"),
    ("Me niegan las vacaciones acumuladas que no tomé en tres años.", "Laboral"),
    # Nuevas
    ("Me pagaron el salario mínimo pero trabajé como supervisor, reclamo diferencia salarial.", "Laboral"),
    ("La empresa me obligó a firmar una renuncia bajo presión y amenazas.", "Laboral"),
    ("Quiero reclamar el auxilio de cesantías e intereses que nunca me consignaron.", "Laboral"),
    ("Me cambiaron el cargo y las funciones sin modificar el salario acordado.", "Laboral"),
    ("Llevo seis meses de incapacidad médica y la empresa quiere terminar mi contrato.", "Laboral"),
    ("El empleador no me dio dotación de trabajo en todo el año.", "Laboral"),
    ("Me despidieron estando en período de prueba sin respetar el preaviso.", "Laboral"),
    ("Quiero reclamar el pago de dominicales y festivos laborados durante tres años.", "Laboral"),
    ("Mi empleador me tiene con contrato de prestación de servicios pero trabajo como empleado.", "Laboral"),
    ("Me liquidaron mal las prestaciones sociales al terminar el vínculo laboral.", "Laboral"),
    ("La empresa entró en liquidación y no sé cómo cobrar mis acreencias laborales.", "Laboral"),
    ("Quiero reclamar indemnización por despido sin justa causa durante período de maternidad.", "Laboral"),
    ("El sindicato fue disuelto ilegalmente y quiero restablecer los derechos colectivos.", "Laboral"),
    ("Mi empleador instaló cámaras en el baño y violó mi intimidad en el trabajo.", "Laboral"),
    ("Me rebajaron el salario unilateralmente sin previo aviso ni acuerdo.", "Laboral"),
    ("Fui despedido por pertenecer a un sindicato y quiero demandarlo por discriminación.", "Laboral"),
    ("No me pagaron la prima de servicios del primer semestre del año.", "Laboral"),
    ("Me obligaron a trabajar más de doce horas diarias sin pago de horas extras.", "Laboral"),
    ("Quiero reclamar el pago de viáticos y gastos de representación no reembolsados.", "Laboral"),
    ("Me negaron la licencia de paternidad a la que tenía derecho por ley.", "Laboral"),

    # ---- DERECHO PENAL (30 muestras) ----
    # Originales
    ("Me robaron el celular y quiero interponer denuncia por hurto.", "Penal"),
    ("Recibí amenazas de muerte por parte de un vecino.", "Penal"),
    ("Fui víctima de estafa en una compra por internet.", "Penal"),
    ("Me acusan falsamente de un delito que no cometí.", "Penal"),
    ("Presencié un homicidio y quiero declarar como testigo.", "Penal"),
    ("Me golpearon en la calle y quiero denunciar lesiones personales.", "Penal"),
    ("Mi ex pareja me está extorsionando con fotos íntimas.", "Penal"),
    ("Sospecho que mi socio está lavando activos en la empresa.", "Penal"),
    ("Me imputaron tráfico de estupefacientes y soy inocente.", "Penal"),
    ("Quiero interponer querella por injuria y calumnia en redes sociales.", "Penal"),
    # Nuevas
    ("Me entraron a robar la casa con violencia y quiero denunciar el hurto agravado.", "Penal"),
    ("Fui víctima de violencia sexual y quiero saber cómo interponer la denuncia.", "Penal"),
    ("Me detuvieron sin orden judicial y quiero interponer habeas corpus.", "Penal"),
    ("Quiero denunciar a mi vecino por maltrato animal contra sus perros.", "Penal"),
    ("Me acusan de peculado y soy funcionario público inocente.", "Penal"),
    ("Un conductor borracho chocó mi carro y huyó, quiero denunciarlo.", "Penal"),
    ("Quiero denunciar una red de trata de personas que descubrí en mi barrio.", "Penal"),
    ("Me falsificaron la firma en un documento y quiero denunciar la falsedad.", "Penal"),
    ("Alguien suplantó mi identidad y sacó créditos a mi nombre.", "Penal"),
    ("Me acusan de violencia intrafamiliar y las pruebas en mi contra son falsas.", "Penal"),
    ("Quiero denunciar a un funcionario que me pidió dinero para agilizar un trámite.", "Penal"),
    ("Me amenazaron con publicar fotos mías si no pago dinero, ¿qué delito es?", "Penal"),
    ("Fui víctima de un atraco a mano armada en el transporte público.", "Penal"),
    ("Quiero denunciar a mi empleador por apropiación de dineros de los trabajadores.", "Penal"),
    ("Me acusan de abuso de confianza por un dinero que me prestaron.", "Penal"),
    ("Quiero denunciar a un médico por negligencia que causó la muerte de mi familiar.", "Penal"),
    ("Fui víctima de phishing bancario y me robaron los ahorros de mi cuenta.", "Penal"),
    ("Me golpeó un policía sin causa justa y quiero denunciar abuso de autoridad.", "Penal"),
    ("Quiero impugnar la preclusión de mi caso porque el fiscal archivó sin investigar.", "Penal"),
    ("Me capturaron por error de identidad y permanecí detenido tres días injustamente.", "Penal"),

    # ---- DERECHO CIVIL (30 muestras) ----
    # Originales
    ("El arrendatario lleva tres meses sin pagar el arriendo.", "Civil"),
    ("Quiero demandar a mi vecino por daños a mi propiedad.", "Civil"),
    ("El vendedor incumplió el contrato de compraventa del inmueble.", "Civil"),
    ("Necesito liquidar la herencia de mis padres fallecidos.", "Civil"),
    ("Mi deudor no paga la deuda y tenemos un pagaré firmado.", "Civil"),
    ("El constructor no entregó el apartamento en los plazos prometidos.", "Civil"),
    ("Quiero impugnar el testamento de mi abuela por vicios de voluntad.", "Civil"),
    ("Tengo un predio y el vecino me tiene bloqueado el paso.", "Civil"),
    ("El banco me cobró intereses usurarios en el crédito hipotecario.", "Civil"),
    ("Quiero demandar por responsabilidad civil extracontractual por accidente.", "Civil"),
    # Nuevas
    ("Necesito disolver y liquidar la sociedad limitada que tengo con mi hermano.", "Civil"),
    ("El arrendador no me devuelve el depósito al terminar el contrato de arrendamiento.", "Civil"),
    ("Quiero iniciar proceso de pertenencia sobre un lote que llevo 20 años poseyendo.", "Civil"),
    ("Firmé un contrato de promesa de compraventa y el vendedor se arrepintió.", "Civil"),
    ("Me vendieron un vehículo con vicios ocultos y el motor estaba averiado.", "Civil"),
    ("Necesito declarar la simulación de un contrato con el que me defraudaron.", "Civil"),
    ("El inquilino dañó gravemente el inmueble y no quiere pagar las reparaciones.", "Civil"),
    ("Quiero registrar una servidumbre de paso en la Oficina de Registro.", "Civil"),
    ("Un contratista realizó una obra defectuosa en mi casa y no responde.", "Civil"),
    ("Necesito acción reivindicatoria para recuperar mi predio que ocupa un tercero.", "Civil"),
    ("Me vendieron un apartamento con hipoteca oculta y sin informármelo.", "Civil"),
    ("Quiero demandar a una empresa de seguros que no pagó el siniestro del carro.", "Civil"),
    ("Necesito acción de tutela para proteger mis derechos de consumidor ante una empresa.", "Civil"),
    ("Me estafaron en un contrato de franquicia y quiero que lo declaren nulo.", "Civil"),
    ("El fideicomiso que administra mis bienes no rinde cuentas hace dos años.", "Civil"),
    ("Mi socio sacó dinero de la empresa sin autorización y quiero demandarlo.", "Civil"),
    ("Quiero ceder mi posición contractual en un arrendamiento comercial.", "Civil"),
    ("El banco ejecutó la garantía hipotecaria sin notificarme previamente.", "Civil"),
    ("Necesito acción popular para proteger el espacio público que invadió un constructor.", "Civil"),
    ("Me negaron la entrega del inmueble que compré de contado hace seis meses.", "Civil"),

    # ---- DERECHO ADMINISTRATIVO (30 muestras) ----
    # Originales
    ("La alcaldía demolió mi negocio sin orden judicial ni compensación.", "Administrativo"),
    ("Me negaron el registro de mi empresa sin justificación.", "Administrativo"),
    ("Quiero interponer acción de tutela contra el ICFES por negación de título.", "Administrativo"),
    ("La entidad pública no me responde la petición que radiqué hace dos meses.", "Administrativo"),
    ("Me sancionaron disciplinariamente de forma injusta en mi cargo público.", "Administrativo"),
    ("El contrato estatal fue adjudicado irregularmente sin cumplir los requisitos.", "Administrativo"),
    ("Quiero impugnar un acto administrativo que me perjudica.", "Administrativo"),
    ("Me negaron la pensión de vejez teniendo todos los requisitos.", "Administrativo"),
    ("El INVIMA retiró mi producto del mercado sin sustento técnico.", "Administrativo"),
    ("Quiero reclamar indemnización al Estado por falla en el servicio médico.", "Administrativo"),
    # Nuevas
    ("La Secretaría de Salud clausuró mi restaurante sin proceso previo ni descargos.", "Administrativo"),
    ("Me negaron la licencia de construcción sin motivar la decisión.", "Administrativo"),
    ("Quiero demandar al Estado por privación injusta de la libertad.", "Administrativo"),
    ("La entidad pública incumplió el contrato estatal y no reconoce los sobrecostos.", "Administrativo"),
    ("Fui retirado del servicio por supresión del cargo sin el debido proceso.", "Administrativo"),
    ("Me negaron el derecho de petición en dos oportunidades consecutivas.", "Administrativo"),
    ("Quiero interponer acción de nulidad contra una resolución de la Superintendencia.", "Administrativo"),
    ("La Procuraduría me abrió investigación disciplinaria con pruebas obtenidas ilegalmente.", "Administrativo"),
    ("Me liquidaron la pensión de jubilación con un salario base inferior al real.", "Administrativo"),
    ("El municipio no me ha pagado las obras ejecutadas en el contrato de infraestructura vial.", "Administrativo"),
    ("Quiero impugnar el resultado de un concurso de méritos en el que fui descalificado.", "Administrativo"),
    ("La CAR me impuso una multa ambiental desproporcionada sin pruebas.", "Administrativo"),
    ("Me negaron el reconocimiento de la pensión de invalidez por riesgo común.", "Administrativo"),
    ("Quiero reclamar reparación directa por los daños que causó una obra pública en mi casa.", "Administrativo"),
    ("La UARIV no me reconoce como víctima del conflicto armado teniendo todos los soportes.", "Administrativo"),
    ("Me excluyeron del Sisbén sin notificación y perdí los subsidios del Estado.", "Administrativo"),
    ("Quiero demandar al ICBF por privación arbitraria de la custodia de mis hijos.", "Administrativo"),
    ("La Contraloría me formuló un pliego de cargos sin pruebas de detrimento patrimonial.", "Administrativo"),
    ("Me negaron la inscripción en el registro de proponentes de la Cámara de Comercio.", "Administrativo"),
    ("Quiero interponer acción de tutela para proteger mi derecho a la salud frente a la EPS.", "Administrativo"),
]

# Convertir a DataFrame
df = pd.DataFrame(datos_juridicos, columns=["texto", "categoria"])

print("=" * 70)
print("  JustIA - Sistema de Clasificación Automática de Textos Jurídicos")
print("=" * 70)
print(f"\n📄 Total de registros en el dataset: {len(df)}")
print(f"📂 Categorías: {df['categoria'].unique().tolist()}")
print(f"\n📊 Distribución por categoría:")
print(df['categoria'].value_counts().to_string())

# ==============================================================================
# 2. CONSTRUCCIÓN DEL PIPELINE DE CLASIFICACIÓN
# ==============================================================================

print("\n" + "=" * 70)
print("  FASE 1: Entrenamiento del Modelo")
print("=" * 70)

# Dividir datos en entrenamiento y prueba
X = df['texto']
y = df['categoria']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n🔀 División de datos:")
print(f"   - Entrenamiento: {len(X_train)} muestras")
print(f"   - Prueba:        {len(X_test)} muestras")

# Pipeline: TF-IDF + Naive Bayes Multinomial
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),      # Unigramas y bigramas
        min_df=2,                 # Término debe aparecer al menos 2 veces
        max_features=2000,        # Vocabulario ampliado para mayor cobertura
        sublinear_tf=True         # Suavizado logarítmico
    )),
    ('clf', MultinomialNB(alpha=0.3))  # Alpha reducido: más datos = menos suavizado necesario
])

# Entrenar
pipeline.fit(X_train, y_train)
print("\n✅ Modelo entrenado exitosamente.")

# ==============================================================================
# 3. EVALUACIÓN DEL MODELO
# ==============================================================================

print("\n" + "=" * 70)
print("  FASE 2: Evaluación del Modelo")
print("=" * 70)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Exactitud (Accuracy): {acc:.2%}")
print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("\n🔢 Matriz de Confusión:")
labels = sorted(df['categoria'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df.to_string())

# ==============================================================================
# 4. FUNCIÓN DE INFERENCIA (Clasificación de nuevos textos)
# ==============================================================================

def clasificar_consulta(texto: str) -> dict:
    """
    Clasifica un texto jurídico en una de las 5 categorías del sistema.

    Parámetros:
        texto (str): Consulta o fragmento jurídico a clasificar.

    Retorna:
        dict: Categoría predicha y probabilidades por clase.
    """
    categoria = pipeline.predict([texto])[0]
    probs = pipeline.predict_proba([texto])[0]
    clases = pipeline.classes_
    probabilidades = {c: round(float(p), 4) for c, p in zip(clases, probs)}
    return {
        "texto": texto,
        "categoria_predicha": categoria,
        "confianza": round(max(probs) * 100, 1),
        "probabilidades": probabilidades
    }

# ==============================================================================
# 5. DEMOSTRACIÓN CON CASOS DE PRUEBA
# ==============================================================================

print("\n" + "=" * 70)
print("  FASE 3: Demostración con Casos de Prueba Nuevos")
print("=" * 70)

casos_demo = [
    "Mi jefe no me paga el subsidio de transporte y tampoco las cesantías.",
    "Fui víctima de violación de domicilio y quiero denunciarlo.",
    "Necesito disolver la sociedad limitada que tengo con mi hermano.",
    "La DIAN me embargó la cuenta sin notificarme previamente.",
    "Quiero recuperar la custodia de mi hija después de la separación.",
]

print()
for caso in casos_demo:
    resultado = clasificar_consulta(caso)
    print(f"📝 Consulta: \"{resultado['texto'][:70]}...\"" if len(resultado['texto']) > 70 
          else f"📝 Consulta: \"{resultado['texto']}\"")
    print(f"   ➤ Categoría: {resultado['categoria_predicha']} "
          f"(Confianza: {resultado['confianza']}%)")
    top2 = sorted(resultado['probabilidades'].items(), key=lambda x: x[1], reverse=True)[:2]
    print(f"   ➤ Top 2 probabilidades: {top2[0][0]}={top2[0][1]:.4f}, "
          f"{top2[1][0]}={top2[1][1]:.4f}")
    print()

# ==============================================================================
# 6. RESUMEN FINAL
# ==============================================================================

print("=" * 70)
print("  RESUMEN DEL SISTEMA JustIA - MVP")
print("=" * 70)
print(f"""
  Modelo:       TF-IDF + Multinomial Naive Bayes
  Pipeline:     scikit-learn (ngram 1-2, max_features=2000, alpha=0.3)
  Categorías:   Familia, Laboral, Penal, Civil, Administrativo
  Dataset:      150 muestras sintéticas (30 por categoría)
  Exactitud:    {acc:.2%} en conjunto de prueba
  Uso:          clasificar_consulta("texto de la consulta jurídica")

  ⚠️  NOTA ÉTICA: Este sistema es un apoyo a la orientación legal.
      No reemplaza el criterio de un abogado o juez. Toda decisión
      jurídica final debe ser validada por un profesional del derecho.
""")
print("=" * 70)
