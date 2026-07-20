# Seismic BYOL

Repositório experimental para pré-treinamento autossupervisionado com BYOL e
segmentação de fácies sísmicas.

> O código histórico ainda está presente em `scripts/` e
> `dev-seismic-byol/`. A interface descrita abaixo é a fundação reprodutível que
> será usada para migrar esses pipelines gradualmente.

## Fundação de experimentos

A configuração em `configs/experiments/paper_main.yaml` representa o trabalho
como duas matrizes:

1. pré-treinamentos BYOL por dataset e seed;
2. combinações downstream de origem do backbone, dataset, cabeça, política de
   congelamento, quantidade de amostras e seed.

O YAML é declarativo. Nenhuma combinação precisa ser copiada manualmente para
um script.

### Instalação

Somente para validar e planejar experimentos:

```bash
python -m pip install -e .
```

Para executar os pipelines de ML nas próximas etapas:

```bash
python -m pip install -e ".[runtime]"
```

O extra `runtime` fixa o Minerva oficial no commit
`f23918a58ea25ee4c016860a348d592e4ac5a04d` (`0.3.10-beta`). Isso substitui a
dependência implícita de uma pasta `Minerva-Dev` editável e sem versão.

### Validar

```bash
seismic-byol validate configs/experiments/paper_main.yaml
```

Saída esperada:

```text
Valid configuration: paper-main (2 matrices, 3660 runs)
```

### Inspecionar o plano

Por padrão, as primeiras 20 execuções são mostradas:

```bash
seismic-byol plan configs/experiments/paper_main.yaml
```

Filtros são tipados como YAML e podem ser repetidos:

```bash
seismic-byol plan configs/experiments/paper_main.yaml \
  --only matrix=downstream \
  --only pretrain=both_N \
  --only finetune=f3_N \
  --only cap=32 \
  --only seed=2 \
  --limit 0
```

Os formatos `table`, `json` e `yaml` são suportados:

```bash
seismic-byol plan configs/experiments/paper_main.yaml \
  --format json --limit 0
```

### Gerar manifestos resolvidos

```bash
seismic-byol plan configs/experiments/paper_main.yaml \
  --only pretrain=both_N \
  --write-manifests outputs/plans/paper-main
```

O limite de exibição não limita os manifestos escritos. Cada arquivo recebe um
`run_id` determinístico calculado a partir da combinação, parâmetros e revisão
do Minerva.

## Formato da matriz

Um arquivo pode conter várias matrizes, permitindo representar estágios
distintos do mesmo trabalho:

```yaml
schema_version: 1

experiment:
  name: example
  metadata:
    minerva:
      revision: commit-do-minerva

matrices:
  - name: downstream
    stage: finetune
    axes:
      pretrain: [both_N, imagenet, scratch]
      finetune: [f3_N, seam_ai_N]
      seed: [0, 1, 2, 3, 4]
    exclude:
      - pretrain: scratch
        seed: 4
    include:
      - pretrain: custom
        finetune: f3_N
        seed: 10
    parameters:
      minerva:
        pipeline:
          class_path: minerva.pipelines.experiment.Experiment
```

Regras:

- `axes` gera o produto cartesiano;
- `exclude` aceita seletores parciais;
- `include` exige uma combinação completa e pode adicionar valores externos
  aos eixos;
- nomes de matrizes devem ser únicos;
- valores duplicados e filtros desconhecidos são rejeitados;
- todos os valores precisam ser serializáveis em JSON para que o hash seja
  estável.

## Integração com Minerva

Esta camada não reimplementa treino, datasets ou métricas. O YAML usa a
convenção `class_path`/`init_args` do `jsonargparse` adotada pelo Minerva e
referencia seus componentes:

- `BYOL` e `DeepLabV3Backbone`;
- `DeepLabV3`;
- `TiffReader`, `PNGReader` e `SimpleDataset`;
- `MinervaDataModule` e `BinaryTreeSubset`;
- `ContrastiveTransform` e demais transforms;
- `SimpleLightningPipeline` e `Experiment`.

A próxima etapa conectará cada manifesto resolvido a esses objetos e aos
datasets dos três ambientes. Até lá, os comandos `validate` e `plan` não
inicializam CUDA nem importam PyTorch/Minerva, portanto podem ser usados em
máquinas leves e em CI.

## Testes

Os testes da fundação usam apenas a biblioteca padrão e PyYAML:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```