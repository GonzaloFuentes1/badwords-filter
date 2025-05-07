import argparse
import json
import multiprocessing as mp
import os
import re
from time import time

from datasets import Dataset, load_from_disk
from tqdm import tqdm


# ==== Inicializador de workers ====
def init_worker(bad_words, column_name):
    global patron, text_column
    patron = re.compile(r'\b(' + '|'.join(map(re.escape, bad_words)) + r')\b',
                        re.IGNORECASE)
    text_column = column_name

# ==== Procesamiento individual ====


def procesar_ejemplo(example):
    texto = example[text_column]
    match = patron.search(texto)
    if not match:
        return example
    return None

# ==== Función principal ====


def main(args):
    print(f"📂 Cargando dataset desde: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    if args.text_column not in dataset.column_names:
        raise ValueError(f"La columna '{args.text_column}' no existe en el dataset.")

    with open(args.bad_words_path, 'r', encoding='utf-8') as file:
        palabras = list(map(str.lower, file.read().splitlines()))

    N = min(args.num_docs, len(dataset))
    subset = dataset.select(range(N))

    num_workers = args.num_workers if args.num_workers else mp.cpu_count()
    chunksize = max(1, N // (4 * num_workers))

    print(f"🚀 Procesando {N} documentos con {num_workers} procesos en la columna "
          f"'{args.text_column}'...")
    start_time = time()

    with mp.Pool(num_workers, initializer=init_worker,
                 initargs=(palabras, args.text_column)) as pool:
        mapped_results = pool.imap(procesar_ejemplo, subset, chunksize=chunksize)
        resultados = list(filter(None, tqdm(mapped_results, total=N)))

    end_time = time()
    print(f'✅ Tiempo total: {end_time - start_time:.2f} segundos')
    print(f'🔍 {len(resultados)} coincidencias encontradas')

    if not resultados:
        print("⚠️ No se encontraron coincidencias. No se generará salida.")
        return

    # === GUARDADO DE RESULTADOS ===
    if args.output_mode == "json":
        save_path = args.save_path or os.path.join(args.dataset_path, "filtered_results.json")
        print(f"Guardando resultados como JSON en: {save_path}")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)

    elif args.output_mode == "dataset":
        save_path = args.save_path or os.path.join(args.dataset_path, "filtered")
        print(f"Guardando resultados como Dataset en: {save_path}")
        ds_resultado = Dataset.from_list(resultados)
        ds_resultado.save_to_disk(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filtro de palabras prohibidas en un dataset.")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Ruta al dataset (formato Hugging Face).")
    parser.add_argument('--bad_words_path', type=str, required=True,
                        help="Ruta al archivo de palabras prohibidas.")
    parser.add_argument('--text_column', type=str, default="texto",
                        help="Nombre de la columna con el texto a analizar.")
    parser.add_argument('--num_docs', type=int, default=1000000,
                        help="Cantidad de documentos a procesar.")
    parser.add_argument('--num_workers', type=int, default=None,
                        help="Número de procesos paralelos (por defecto usa todos).")
    parser.add_argument('--output_mode', type=str, choices=["json", "dataset"],
                        default="dataset", help="Modo de salida: 'json' o 'dataset'.")
    parser.add_argument('--save_path', type=str, default=None,
                        help="Ruta personalizada para guardar los resultados.")
    args = parser.parse_args()
    main(args)
    args = parser.parse_args()
    main(args)
