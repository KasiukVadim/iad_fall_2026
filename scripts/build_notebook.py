"""Create and execute the explanatory notebook using reusable modules."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook();cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# Maxwell: EM scattering and tag-reader coupling

Лабораторная работа • 3D + время • символьные и численные модели

**Вопрос:** как связать символьную запись Максвелла, граф вычислений, решение PDE и идентификацию параметров редуцированной модели?

Физические блоки демонстрируют разные режимы электродинамики. Оптические константы нельзя переносить на 13.56 MHz. Численные эксперименты не воспроизводят геометрию антенны из статьи.''')
code('''from pathlib import Path
import sys,json
ROOT=Path.cwd()
if ROOT.name=='notebooks': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import sympy as s
from IPython.display import display,Image,Markdown
from emlab.symbolic import *
from emlab.scattering import *
from emlab.coupling import *
metrics=json.loads((ROOT/'results/metrics.json').read_text())
def show(name): display(Image(filename=str(ROOT/'results'/f'{name}.png')))
''')
md('''## 1. Журнальные источники и постановка

- Sharif et al., Scientific Reports (2023), DOI **10.1038/s41598-023-39825-9**: Maxwell-поля и эквивалентная схема связи reader/tag. Используем идею двух связанных контуров, а не спорные размерности отдельных напечатанных формул.
- Abdelrahman et al., Scientific Reports (2017), DOI **10.1038/s41598-017-15192-0**: рассеяние на сфере, коэффициенты Ми.
- Malitson (1965) и Daimon & Masumura (2007): два независимых исследования дисперсии материалов, HTTP-копии из RefractiveIndex.INFO.

Внешние файлы содержат опубликованные дисперсионные аппроксимации, а не сырые измерения 3D-полей. Их назначение — вариативность материала в модели рассеяния. Данные RFID для подбора параметров синтетические.''')
code("display(json.loads((ROOT/'data/provenance.json').read_text()))")
md(r'''## 2. Символьные модели

$$\epsilon\partial_t\mathbf E+\sigma\mathbf E+\mathbf J-\nabla\times\mathbf H=0,\qquad
\mu\partial_t\mathbf H+\nabla\times\mathbf E=0.$$
$$\nabla\cdot\mathbf D=\rho,\quad \nabla\cdot\mathbf B=0.$$

Компоненты зависят от $(x,y,z,t)$. Запись выше предполагает постоянные $\epsilon,\mu$; неоднородный численный солвер ниже использует $\mathbf D=\epsilon(\mathbf x)\mathbf E$.

Сравниваем SymPy Matrix (явные компоненты), sympy.vector (геометрические операторы) и SymEngine (независимый движок алгебры). Времена единичных вызовов не являются репрезентативным рейтингом библиотек.''')
code("for name,eq in models().items():\n    display(Markdown(f'**{name}**')); display(eq)\ndisplay(compare_backends())")
md('''## 3. Отображения полей и тождества

Градиент: R³→R³ для скалярного поля; ротор: векторное поле→векторное поле; дивергенция возвращает скаляр. Дополнительно: скалярное/векторное произведение, матрица Якоби, векторный лапласиан.''')
code('''f=x*x*y+s.sin(z)
F=s.Matrix([y*z,x*x*z,x*y*y]);G=grad(f)
display({'gradient':G,'curl':curl(F),'divergence':div(F),'dot':F.dot(G),
         'cross':F.cross(G),'Jacobian':F.jacobian(XYZ),
         'vector Laplacian':s.Matrix([div(grad(v)) for v in F])})
assert curl(grad(f))==s.zeros(3,1)
assert s.simplify(div(curl(F)))==0
print('curl(grad f)=0; div(curl F)=0: verified')''')
md('''## 4. Словарь, граф выражения, матрица смежности

`equations.json` хранит каноническую запись `srepr`. Для одного компонентного уравнения строится дерево вхождений AST: ребро parent→operand. Повторные выражения сохраняют разные ID; подписи и порядок узлов записаны отдельно. Это граф вычисления выражения, а не пространственная сетка и не причинный граф.''')
code("show('expression_graph');show('adjacency')")
md(r'''## 5. Рассеяние: точная модель и аппроксимация

Немагнитная сфера радиуса $a$ в вакууме:
$$Q_{sca}=\frac{2}{x^2}\sum_{n=1}^{N}(2n+1)(|a_n|^2+|b_n|^2),\quad x=2\pi a/\lambda.$$
В пределе $x\ll1$:
$$Q_R=\frac83x^4\left|\frac{m^2-1}{m^2+2}\right|^2.$$

Волны монохроматические: физическое поле восстанавливается как Re[E(r) exp(-iωt)]. Фазовая конвенция не меняет Q. Используем прозрачные материалы и диапазон 0.45–1 μm, радиус 25 nm. CSV содержит n(λ), Ми и Рэлей.''')
code("show('materials_scattering');show('rayleigh_error')\ndisplay({k:v for k,v in metrics.items() if 'rayleigh' in k})")
md(r'''## 6. Полное 3D-решение Максвелла во времени

Нормированные единицы $c=\epsilon_0=\mu_0=1$, $\Omega=[0,1)^3$, периодические границы:
$$\dot{\mathbf D}=\nabla_h\times\mathbf H,\qquad
\dot{\mathbf H}=-\nabla_h\times(\mathbf D/\epsilon(\mathbf x)).$$

Начальные данные $D_y=H_z=\cos(2\pi x)$, остальные компоненты нулевые. Материал: сглаженная сфера $\epsilon=1+1.1\exp[-(r/0.14)^8]$. Шаги: центральные разности O(h²), RK4, 24³ узла, 160 шагов до t=0.65.

Плоская начальная волна и сферическая неоднородность формируют поле, зависящее от трёх пространственных координат. Разность с однородным IVP показывает возмущение; начальные E внутри включения различаются. Это **периодическая начальная задача**, а не изолированная сфера с условием излучения. Её нельзя сравнивать напрямую с far-field Q из Ми.''')
code("show('maxwell_slices');show('maxwell_3d');display(Image(filename=str(ROOT/'results/maxwell_evolution.gif')))\nf=np.load(ROOT/'results/fields_3d.npz')\nprint('Stored vector fields:',f['E'].shape,'time frames and 3 components on 3D grid')")
md('''## 7. Проверка решения

Смотрим дискретные div D, div H и дрейф энергии. Для однородной среды отдельно сравниваем с точной плоской волной cos[2π(x−t)] на сетках 12³, 24³, 48³. Это количественная проверка порядка схемы; гладкая неоднородная сфера не имеет здесь независимого точного решения.''')
code("show('conservation');show('convergence')\ndisplay({k:v for k,v in metrics.items() if k.startswith(('maxwell','plane_wave'))})")
md(r'''## 8. Геометрия reader/tag → взаимная индуктивность

$$M=\frac{\mu_0}{4\pi}\oint\oint\frac{d\boldsymbol\ell_1\cdot d\boldsymbol\ell_2}{|\mathbf r_1-\mathbf r_2|}.$$

Две круговые катушки: 40 и 15 mm, расстояние 40 mm, воздушная среда, один виток. Интеграл Неймана вычисляется квадратурой по 240 точкам. Можно менять положение и наклон в 3D. При 13.56 MHz длина волны около 22 m: размеры намного меньше λ. Собственные L задаются отдельно; формула для тонких пересекающихся проводов неприменима.''')
code("show('coils_3d');show('coupling_map')")
md(r'''## 9. Уточнение параметров модели-решения ODE

$$\begin{bmatrix}L_1&M\\M&L_2\end{bmatrix}\dot{\mathbf i}
+\operatorname{diag}(R_1,R_2)\mathbf i+
\operatorname{diag}(1/C_1,1/C_2)\mathbf q=\begin{bmatrix}V_s\\0\end{bmatrix},\quad\dot{\mathbf q}=\mathbf i.$$

$L_1=1$ μH, $L_2=0.5$ μH, $R_1=2$ Ω, $R_2=4$ Ω. C настроены на 13.56 MHz. Начальные токи и напряжения нулевые; Vs=cos(2πft) V.

Вектор θ=(M [nH], R₂ [Ω]); начальная точка (12,6), bounds ([0.1,0.5],[100,15]). Ожидаемые значения: M из геометрии, R₂=4 Ω. Пассивность: M²<L₁L₂. DOP853 решает ODE, least_squares минимизирует нормированные остатки.

Шум Gaussian, σ=1% peak current, seed фиксирован. Каждая третья точка исключена из fit; held-out RMSE считается относительно известной чистой траектории. Это интерполяционная проверка на той же траектории, не проверка на новом устройстве.''')
code("show('parameter_fit');display(metrics['coupling_fit'])")
md('''## 10. Смешанные модели

`HybridModel` объединяет геометрию M(d, tilt), добавочную функцию ΔM(features) и физический ODE-солвер. По умолчанию поправка нулевая. Это интерфейс для будущей обучаемой поправки; нейросеть не обучена и улучшение качества не заявлено. Пассивность проверяется после поправки.''')
code("hybrid=HybridModel();tt=np.linspace(0,1e-7,20);print('Hybrid output shape:',hybrid.predict(tt).shape)")
md('''## 11. Выводы и ограничения

1. Символьная модель сохранена как словарь, AST-граф и матрица смежности; проверены векторные тождества.
2. Рэлей проверен против Ми, а не выдан за точное решение при произвольном размере сферы.
3. 3D Maxwell IVP проверен по ограничениям, энергии и сходимости плоской волны.
4. M и R₂ восстановлены по синтетическому переходному процессу; значения и ошибки находятся выше.
5. Реализован интерфейс физической модели с корректирующей функцией.

**Что потребует данных преподавателя:** если нужны именно два набора экспериментальных 3D+time полей или реальные reader/tag измерения, опубликованные дисперсионные коэффициенты и синтетическая траектория это требование не заменяют. В приложенном PDF конкретные датасеты для этой темы не указаны.

Тесты: `python -m pytest -q`. Воспроизведение: см. README. Использование ИИ раскрыто в `docs/AI_USAGE.md`.''')
nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
path=ROOT/'notebooks/maxwell_lab.ipynb';nbf.write(nb,path)
if '--in-process' in __import__('sys').argv:
    # This environment forbids Jupyter TCP sockets. Execute real cells in order
    # inside IPython and preserve rich output, without launching a kernel.
    import os
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    os.chdir(ROOT);shell=InteractiveShell.instance();count=0
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output() as captured:
            result=shell.run_cell(cell.source,store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Cell {count} failed') from (result.error_before_exec or result.error_in_exec)
        outputs=[]
        if captured.stdout:outputs.append(nbf.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:outputs.append(nbf.v4.new_output('stream',name='stderr',text=captured.stderr))
        for o in captured.outputs:
            outputs.append(nbf.v4.new_output('display_data',data=o.data,metadata=o.metadata))
        cell.outputs=outputs;cell.execution_count=count
    nb.metadata['execution']={'method':'IPython in-process; network sockets unavailable'}
else:
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
nbf.write(nb,path);print(path)
