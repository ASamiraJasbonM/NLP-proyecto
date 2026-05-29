import pandas as pd
from typing import List, Tuple


def cargar_datos_sinteticos() -> pd.DataFrame:
    from src.config import CATEGORIAS

    datos: List[Tuple[str, str]] = [
        (
            "Solicito custodia compartida de mis hijos menores tras el divorcio.",
            "Familia",
        ),
        ("Mi esposa pide alimentos para los niños, ¿cuánto debo pagar?", "Familia"),
        ("Quiero iniciar proceso de divorcio por mutuo acuerdo.", "Familia"),
        (
            "Necesito reconocer la paternidad de mi hijo mediante prueba de ADN.",
            "Familia",
        ),
        (
            "Mi ex pareja no me permite ver a mis hijos en las fechas acordadas.",
            "Familia",
        ),
        ("Solicito medida de protección por violencia intrafamiliar.", "Familia"),
        ("Deseo adoptar un menor que está en situación de abandono.", "Familia"),
        (
            "Mi cónyuge oculta bienes durante el proceso de liquidación de sociedad conyugal.",
            "Familia",
        ),
        ("Necesito asesoría para capitulaciones matrimoniales.", "Familia"),
        (
            "Reclamo alimentos para mi madre adulta mayor que no puede trabajar.",
            "Familia",
        ),
        ("Me despidieron sin justa causa después de 5 años en la empresa.", "Laboral"),
        ("No me han pagado las horas extras del último mes.", "Laboral"),
        ("Mi empleador no me quiere afiliar a salud y pensión.", "Laboral"),
        (
            "Sufrí un accidente de trabajo y la ARL no reconoce la incapacidad.",
            "Laboral",
        ),
        ("Me acosan laboralmente y el jefe me presiona para renunciar.", "Laboral"),
        ("No me pagaron las prestaciones sociales al terminar el contrato.", "Laboral"),
        (
            "Quiero demandar a mi empresa por despido discriminatorio por embarazo.",
            "Laboral",
        ),
        (
            "Mi contrato a término fijo no fue renovado y tengo fuero sindical.",
            "Laboral",
        ),
        ("El empleador me descontó el salario sin autorización.", "Laboral"),
        ("Me niegan las vacaciones acumuladas que no tomé en tres años.", "Laboral"),
        ("Me robaron el celular y quiero interponer denuncia por hurto.", "Penal"),
        ("Recibí amenazas de muerte por parte de un vecino.", "Penal"),
        ("Fui víctima de estafa en una compra por internet.", "Penal"),
        ("Me acusan falsamente de un delito que no cometí.", "Penal"),
        ("Presencié un homicidio y quiero declarar como testigo.", "Penal"),
        ("Me golpearon en la calle y quiero denunciar lesiones personales.", "Penal"),
        ("Mi ex pareja me está extorsionando con fotos íntimas.", "Penal"),
        ("Sospecho que mi socio está lavando activos en la empresa.", "Penal"),
        ("Me imputaron tráfico de estupefacientes y soy inocente.", "Penal"),
        (
            "Quiero interponer querella por injuria y calumnia en redes sociales.",
            "Penal",
        ),
        ("El arrendatario lleva tres meses sin pagar el arriendo.", "Civil"),
        ("Quiero demandar a mi vecino por daños a mi propiedad.", "Civil"),
        ("El vendedor incumplió el contrato de compraventa del inmueble.", "Civil"),
        ("Necesito liquidar la herencia de mis padres fallecidos.", "Civil"),
        ("Mi deudor no paga la deuda y tenemos un pagaré firmado.", "Civil"),
        ("El constructor no entregó el apartamento en los plazos prometidos.", "Civil"),
        ("Quiero impugnar el testamento de mi abuela por vicios de voluntad.", "Civil"),
        ("Tengo un predio y el vecino me tiene bloqueado el paso.", "Civil"),
        ("El banco me cobró intereses usurarios en el crédito hipotecario.", "Civil"),
        (
            "Quiero demandar por responsabilidad civil extracontractual por accidente.",
            "Civil",
        ),
        (
            "La alcaldía demolió mi negocio sin orden judicial ni compensación.",
            "Administrativo",
        ),
        ("Me negaron el registro de mi empresa sin justificación.", "Administrativo"),
        (
            "Quiero interponer acción de tutela contra el ICFES por negación de título.",
            "Administrativo",
        ),
        (
            "La entidad pública no me responde la petición que radiqué hace dos meses.",
            "Administrativo",
        ),
        (
            "Me sancionaron disciplinariamente de forma injusta en mi cargo público.",
            "Administrativo",
        ),
        (
            "El contrato estatal fue adjudicado irregularmente sin cumplir los requisitos.",
            "Administrativo",
        ),
        ("Quiero impugnar un acto administrativo que me perjudica.", "Administrativo"),
        (
            "Me negaron la pensión de vejez teniendo todos los requisitos.",
            "Administrativo",
        ),
        (
            "El INVIMA retiró mi producto del mercado sin sustento técnico.",
            "Administrativo",
        ),
        (
            "Quiero reclamar indemnización al Estado por falla en el servicio médico.",
            "Administrativo",
        ),
    ]

    df = pd.DataFrame(datos, columns=["texto", "categoria"])
    return df


def cargar_datos_desde_csv(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta)
