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
# 1. DATOS JURÍDICOS SINTÉTICOS (contexto colombiano)
# ==============================================================================

datos_juridicos = [
    # ---- DERECHO DE FAMILIA ----
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

    # ---- DERECHO LABORAL ----
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

    # ---- DERECHO PENAL ----
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

    # ---- DERECHO CIVIL ----
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

    # ---- DERECHO ADMINISTRATIVO ----
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
        min_df=1,                 # Frecuencia mínima de término
        max_features=500,         # Vocabulario máximo
        sublinear_tf=True         # Suavizado logarítmico
    )),
    ('clf', MultinomialNB(alpha=0.5))  # Suavizado de Laplace
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
  Pipeline:     scikit-learn (ngram 1-2, max_features=500, alpha=0.5)
  Categorías:   Familia, Laboral, Penal, Civil, Administrativo
  Dataset:      50 muestras sintéticas (10 por categoría)
  Exactitud:    {acc:.2%} en conjunto de prueba
  Uso:          clasificar_consulta("texto de la consulta jurídica")

  ⚠️  NOTA ÉTICA: Este sistema es un apoyo a la orientación legal.
      No reemplaza el criterio de un abogado o juez. Toda decisión
      jurídica final debe ser validada por un profesional del derecho.
""")
print("=" * 70)
