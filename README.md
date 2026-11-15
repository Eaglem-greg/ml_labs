# Machine Learning Labs

Учебные проекты по машинному обучению. Каждая тема находится в собственной папке внутри `labs/`; связанные исходники, ноутбуки и данные хранятся вместе.

## Структура

```text
ml_labs_repo/
├── labs/
│   ├── 01_eda_online_retail/
│   │   ├── data/
│   │   └── eda.ipynb
│   ├── 02_linear_regression/
│   │   ├── linear_regression.py
│   │   └── test.ipynb
│   ├── 03_polynomial_regression/
│   │   └── polynomial_regression.py
│   ├── 04_logistic_regression/
│   │   └── logistic_regression.py
│   └── 05_svm/
│       └── svm.py
├── .gitignore
└── README.md
```

## Как добавить новый проект

1. Перейдите в основной локальный репозиторий:

   ```bash
   cd ~/machine_learning_labs/ml_labs_repo
   ```

2. Перед началом работы получите последние изменения:

   ```bash
   git pull --rebase origin main
   ```

3. Создайте следующую папку в `labs/`. Используйте формат `NN_короткое_название`, например:

   ```bash
   mkdir -p labs/06_decision_trees
   ```

4. Поместите в неё все файлы этой работы. Рекомендуемый шаблон:

   ```text
   labs/06_decision_trees/
   ├── README.md          # цель, запуск и результаты
   ├── solution.ipynb     # или основной Python-файл
   ├── src/               # дополнительный код
   ├── tests/             # тесты
   └── data/              # только небольшие данные
   ```

5. Проверьте, что виртуальное окружение, кэш, секреты и временные файлы не попали в коммит:

   ```bash
   git status --short
   git check-ignore -v .venv __pycache__ .env
   ```

6. Добавьте, проверьте и закоммитьте изменения:

   ```bash
   git add labs/06_decision_trees README.md
   git diff --cached --stat
   git commit -m "Add decision trees lab"
   ```

7. Отправьте коммит на GitHub:

   ```bash
   git push origin main
   ```

Если прямой push в `main` запрещён, используйте отдельную ветку:

```bash
git switch -c lab/06-decision-trees
git push -u origin lab/06-decision-trees
```

После этого создайте Pull Request из `lab/06-decision-trees` в `main` на GitHub.

## Правила структуры

- Один учебный проект — одна папка в `labs/`.
- Не создавайте новый `.git` внутри папки проекта: весь каталог `ml_labs_repo` является единым репозиторием.
- Не добавляйте `.venv`, `__pycache__`, `.DS_Store`, `.env` и другие локальные файлы.
- Данные больше 100 МБ храните через Git LFS или во внешнем хранилище; в проекте оставляйте инструкцию для загрузки.
- Если ноутбук использует относительные пути, храните его данные рядом с ним, например в `data/`.
