# 📈 AI Recruiter

AI Recruiter: Otimizando o Match entre Candidatos e Vagas com Machine Learning.

---

## 🎯 Problema

A Decision enfrenta dificuldade para:

- Padronizar entrevistas e armazenar informações valiosas.
- Identificar engajamento real de candidatos.
- Repetir o padrão de sucesso de candidatos bem alocados.

---

## 💡 Solução Proposta

Desenvolver um sistema de IA híbrido, com:

- Pipeline de Machine Learning supervisionado: para prever a probabilidade de sucesso de um candidato com base em no job description da vaga e informacoes do curriculo do profissional
- API de prediction, servindo o modelo em produção.
- Docker para empacotamento.
- Monitoramento de drift, log de previsões e dashboard básico.


---

## ⚙️ Pipeline Técnico
### 📂 1. Coleta & Pré-processamento

Base de dados: Base fornecida pela Decision.

Tratamento: Limpeza de valores ausentes, normalização de variáveis numéricas, encoding de variáveis categóricas.

Feature Engineering:

- Extração de palavras-chave de entrevistas transcritas com NLP (se houver).
- Similaridade do coseno (match_score)

### 🤖 2. Modelagem
Algoritmo supervisionado: Random Forest.

- Target: candidato_aprovado (1) ou não aprovado (0)  {prediction}

Validação: métricas de precisão, recall e F1-score.

Serialização: pickle.

### 🚀 3. Deployment

API: Flask com rota /predict
- Entrada: JSON com atributos do candidato e da vaga.
- Saída: Probabilidade de aprovação.

Dockerfile: Empacotamento da API + dependências.

Deploy: Local

### 🧪 4. Testes

Testes unitários:
- Predição.
- Endpoint da API.

Testes de integração:

- Validação com Postman ou cURL.

### 🔍 5. Monitoramento

- A API expõe um painel interativo em /dashboard com estatísticas de tempo de resposta, chamadas e erros.

- O MLflow é usado para rastrear todas as execuções de treinamento e inferência.


---

## 📁 Estrutura do Projeto

```plaintext
├── api/
├──── app.py # API para predição
├── notebooks/
├──── data/ # Dados para treinamento do modelo
├──── saved_models/ # Modelos e scaler salvos
├──── decision-recruitment-process-model-training.ipynb # Notebook para treino e deploy do modelo
├── requirements.txt
├── Dockerfile
├── start.sh
└── README.md
```

---

## 🚀 Como Rodar Localmente

### 1. Clone o repositório

```bash
git clone https://github.com/diegoalber1/ml-tech-fiap-datathon-decision.git
cd ml-tech-fiap-datathon-decision
```

### 2. Crie e ative o ambiente virtual (opcional)

```bash
python -m venv venv
source venv/bin/activate 
# Windows: venv\Scripts\activate
```
### 3. Instale as dependências

```bash
pip install -r requirements.txt
```
## 📊 Treinar o Modelo

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/decision-recruitment-process-model-training.ipynb
```
## 🔁 Fazer Predições com a API

### 1. Rodar a API localmente

```bash
python api/app.py

mlflow ui --host 0.0.0.0 --port 5001
```
Acesse a API em http://localhost:5000

Dashboard de monitoramento:

http://localhost:5000/dashboard

Mlops UI:

http://localhost:5001

### 2. Exemplo de requisição com curl:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
        {
          "job_description": "Key skills required for the job are:\n\nPeopleSoft Application Engine-L1 (Mandatory)\nHTML 5-L1\n\nAs a Domain Consultant in one of the industry verticals, you are responsible for implementation of roadmaps for business process analysis, data analysis, diagnosis of gaps, business requirements and functional definitions, best practices application, meeting facilitation, and contributes to projectplanning. You are expected to contribute to solution building for the client and practice. Should be able to handle higher scale and complexity and proactive in client interactions.\n\nMinimum work experience:5 - 8 Years\n\nProficiency in English Language is Desirable O recurso Peoplesoft tem como responsabilidades: projetar, desenvolver e testar as altera絥s do Peoplesoft nos ",

          "cv_text": "\n\nárea de atuação: lider de consultoria / gerenciamento de projeto / liderança técnica / analista senior\nresumo de qualificações\n- profissional com 30 anos de experiência na área de ti, adquirida em projetos desenvolvidos em empresas nacionais de grande porte e consultorias especializadas.\n- sólida experiência na área de tecnologia da informação atuando como consultor pleno-senior, liderando equipes de desenvolvimento, implantação e upgrade de sistemas erp, crm, hcm e campus solution.\n- sólida experiência em implantação de novas funcionalidades e backoffice à solução peoplesoft.\n- forte atuação junto a clientes internos e externos para implantação de novos processos de negócio através de soluções erp – crm e hcm.\n- vivência nos processos de integração entre módulos do peoplesoft e sistemas externos e internos.\n- experiência em ferramentas para gerenciamento de projetos (msproject e hp ppm).\n- experiência em consultoria remota e monitoramento de processos de produção à distância.\n- vivência nos processos de análise e levantamento de requisitos, detalhamento de escopo, definição de arquitetura, elaboração de especificações técnicas/funcionais e implantação de processos de produção.\n- experiência na administração do ambiente e nos processos de administração de acessos e perfis.\n- sólida experiência no suporte a produção, administração de chamados e riscos, apresentação de scorecard semanal e realização de treinamento.\n\n\nformação\n- graduação – universidade castelo branco – tecnólogo em processamento de dados – 1995\n\n\ncapacitação técnica\n- peoplesoft (peopletools 8.49, 8.53, 8.55, 8.57 peoplecode, sqr, crystal, xml publisher, query manager, app engine, app package, process scheduler, segurança, app message, integration broker, workflow, tree manager)\n- peoplesoft fscm 8.9, 9.1 hcm 9.0, 9.2, crm e cs\n- microsoft visual basic, java, xml, javascript, html, css, jquery, python\n- ios, windows, linux, unix\n- microsoft sql server, microsoft access, mysql, oracle.\n- oracle pl/sql, procedures, triggers.\n- oracle oic\n\n\ncursos de extensão\n- microsoft visual basic – nsi – 1995\n- trailhead playground management – trailhead by salesforce – dez-2019\n- noções básicas do trailhead – trailhead by salesforce – dez-2019\n- data modeling – trailhead by salesforce – dez-2019\n- salesforce platform basics – trailhead by salesforce – jan-2020\n- crm for salesforce classic - trailhead by salesforce – jan-2020\n\n\nidiomas\n- inglês – avançado\n- espanhol – intermediário\n- português - nativo\n\ncertificados\n- capacitação peoplesoft (formação técnica) – peoplesoft -1999\n- segurança da informação – brasilcap – 2009\n- ad (administração de dados) brasilcap– 2010\n- oracle hcm cloud 2019 sales specialist assessment – ago-2020\n- oracle hcm cloud solution engineer specialist – jul-2020\n- scrum foundation professional certificate (spfc) – certiprof – jul-2020\n- oracle integration cloud services oic – oracle ics – udemy – dez-2020\n\n\nprincipais projetos\n- implementação da suíte financeira do peoplesoft erp na mbr – bh (atual vale), atuando diretamente nos módulos: ap, ar e tr como consultor técnico sênior.\n- implantação da solução oracle peoplesoft erp e crm na aga linde health care - rj, atuando na frente técnica, parametrizando, customizando e desenvolvendo interfaces.\n- implantação de solução oracle peoplesoft erp na porto seguro seguros – sp, atuando diretamente no módulo ap como consultor sênior e líder de equipe. atuei diretamente na construção das interfaces entre o peoplesoft e o sistema legado.\n- implantação da solução oracle peoplesoft na tavex – sp, atuando diretamente como consultor técnico sênior.\n- implantação da resolução 533 da anatel no peoplesoft crm 2012 na claro – sp.\n- implantação de interface na vale (escritório: piura - peru) entre o peoplesoft hcm e o adam (sistema de folha de pagamento) utilizado no peru.\n- implantação peoplesoft hcm 92 na bbts - rj\n- projeto cobrança registrada (peoplesoft fscm – ar) realizado na sulamérica – rj\n- sustentação peoplesoft hcm 92 na vale brasil – rj\n- rollout oracle peoplesoft fscm 91 na cbre brasil - sp\n\n\nexperiência profissional\ndbs-digital– jan/2021\nprojeto laureate – criação de novas bu gl\n- responsável pelo acompanhamento dos testes em uat1/uat2\n- ajustes técnicos para os chamados abertos em uat1/uat2\n- manutenção em integrações com oic erp cloud/erp on premise\n\n\nataway– fev/2020 – dez/2020\nprojeto global cbre – implantação do peoplesoft fin 91 – projeto remoto\n- responsável pelo módulo de compras (po)\n- levantamento dos requisitos\n- workshop das funcionalidades\n- levantamento de integrações entre peoplesoft e coupa system\n- testes unitários e testes integrados\n- treinamento\n\nprojeto br distribuidora – integração oracle oic– projeto remoto\n- manutenção em integrações hcm cloud/erp\n- criação de novas integrações. mapeamentos de dados e regras de conversão\n\n\n\ngrupo quanam– jun/2019 – dez/19\nprojeto revitalização peoplesoft fin 9.1– santista - sp\n- levantamento funcional/técnico de um módulo adicional (cotton) a ser incorporado ao peoplesoft.\neste módulo contempla todo o fluxo de compra e venda de algodão. a versão anterior havia sido escrito em clipper.\n- atuação técnica nos módulos: contas a pagar, compras, inventário e fretes\n- estimativa de tempo baseado em planilha excel com valores padrão para todos os tipos de objetos peoplesoft para gaps novos e gaps de manutenção\n- conversão de processos antigos sqr em novos processos app. engine utilizando app packages\n- conversão de relatórios crystal reports em xml publisher\n- manutenção de rotinas em pl/sql\n\n\noracle do brasil – abr/2017 – maio-2019\nsustentação pós-implementação peoplesoft hcm 92 na vale (brasil) – rj\n- auxilio na comunicação entre o cliente e o time off-shore (índia).\n- acompanhamento das atividades dos usuários\n- acompanhamento das definições de novas atividades (melhorias)\n- levantamento de dados para pmo\n- melhoria das integrações (inbound & outbound) através de webservices e batch process\n- levantamento/analise/solução proposta da migração de sistema externo para dentro do peoplesoft\n- migração de projetos com o peoplesoft cemli manager\n- manutenção de novos projetos com o ppm ( project portfolio management )\n- acompanhamento nas atualizações de sistemas externos e adequação das rotinas pl/sql para atender as novas versões\n- ajuste e monitoramento da interface de folha de pagamento entre psft x fpw\n\n\nstefanini consultoria – jun/2016 – mar/2017\natuando no projeto cobrança registrada na sulamérica – rj\n- realizando tarefas de configuração bancária, mapas bancários, geração de boletos impressos e emissão de boletos online. validando arquivos bancários e preparando arquivos de retorno aos bancos. geração e validação de código de barra e linha digitável\n- manutenção de rotinas em pl/sql (procedures, functions e triggers)\n\n\nf2c (a hitachi consulting company) – jun/2015 – jun/2016\n- atuei no desenvolvimento dos gaps principais (incluindo fluid page) e as interfaces entre o peoplesoft e o ebs, assim como entre o peoplesoft e o sistema legado.\n- construção de rotinas em pl/sql\n\n\ninfosys – mai/2012 a mai/2015\n- suporte a produção para o peoplesoft hcm 9.0 (core hr, benefícios básicos) para a vale peru e chile:\n- atendimento e manutenção de chamados\n- desenvolvimento de melhorias\n- implementação de novas interfaces\n- manutenção em rotinas pl/sql (procedures, functions e triggers)\n\n\n\nlimine solutions – nov/2011 a abr/2012\n- implantação da resolução 533 da anatel no peoplesoft crm 2012 na claro – sp\n- desenvolvimento de melhorias\n- construção de especificação técnica\n- modificação de stored procedures e functions\n- construção dos cenários de teste\n- desenvolvimento, testes unitários, testes de aceite do usuário e migração do projeto para ambiente produtivo\n\n\ncontrato independente – fev/2011 a nov/2011\n- consultor técnico peoplesoft hcm no magazine luiza - sp:\n- desenvolvimento técnico remoto baseado em pacotes de especificação previamente construídas\n- desenvolvimento de novas funcionalidades, testes e migração dos projetos para ambiente de homologação.\n\ngrupo quanam – jan/2006 a nov/2011\nmar/2011 a nov/2011\n- atuei diretamente no suporte a produção em conjunto com a equipe funcional e técnica:\n- construção de novas especificações\n- desenvolvimento e ajustes de interfaces\n- treinamento funcional e técnico\n- manutenção de rotinas em pl/sql\n\npeoplesoft back office na porto seguro seguros – sp (mar/2009 a jun/2009)\nconsultor funcional / técnico:\n- desenvolvimento de novos gaps\n- administração de back office\n- atualização do processo de interface para o contas a pagar\n\nimplementação da solução oracle peoplesoft erp na tavex – sp (jan/2009 a mar/2009)\n- consultor funcional / técnico:\n- desenvolvimento de novos gaps\n- implementação dos módulos: ap, ar, cm, gl, bi, ex, om, po, in e mn\n- construção de relatórios em sqr e crystal report.\n\nimplementação da solução oracle peoplesoft erp na sul américa seguros – rj (jul/2008 a dec/2008)\n- consultor técnico:\n- implementação dos módulos: ap e ar\n- liderança e desenvolvimento de novos gaps\n- construção e manutenção da planilha de atividades\n\n- adaptando e recustomizando gaps da versão anterior\n- validação das definições de projeto\n- migração das definições de projeto\n- export/import de dados via data mover\n- criação e manutenção da planilha de trabalho\n\nimplantação do peoplesoft financial 8.20 na porto seguro seguros – sp (segunda fase) (jan/2006 a dez/2007)\n- consultor funcional/técnico e líder técnico:\n- construção de especificações funcionais e técnicas para os gaps\n- construção de especificações funcionais e técnicas para as interfaces\n- desenvolvimento de gaps\n- desenvolvimento de interfaces entre peoplesoft e sistema legado\n- construção da documentação técnica\n- capacitação de novos consultores\n- treinamento do usuário nos processos de interface\n\n\nbrazilcap - rj – jul/2009 a fev/2011\n- suporte funcional e técnico ao ambiente de produção e implementação de novas melhorias. revisão e reconstrução dos processos críticos do peoplesoft erp e hr:\n- revisão dos processos de produção\n- suporte as demandas de produção\n- construção e desenvolvimento de especificações e gaps para os módulos: ap, gl, am e hr\n- coordenação interna de treinamento técnico para a equipe técnica e novos empregados nos processos peoplesoft.\n- confeccionar especificações para a implementação do sap e desligamento do peoplesoft\n- construção de um novo processo de interface para o ap usando java\n- administração de entrevistas com consultorias externas para a implementação do sap\n\n\nogeda it solutions – sp – jul/2004 a ago/2005\n- primeira fase da implementação do peoplesoft financial 8.20 na porto seguro seguros – sp. substituição do sistema legado construído em 4gl pelo oracle peoplesoft:\n- identificação de novas funcionalidades\n- construção de especificações funcionais e técnicas para os gaps\n- construção de especificações funcionais e técnicas para as interfaces\n- desenvolvimento de gaps\n- desenvolvimento de interfaces entre peoplesoft e sistema legado\n\n\natos origin/peoplefriend – rj – dez/2003 a jun/2004\n- suporte a produção na shell brasil:\n- suporte as demandas de produção\n- extração e importação de dados com o data mover para o peoplesoft hr\n- desenvolvimento de novos gaps\n- extração de dados do peoplesoft para a base de dados do sap\n\n\nadoption/hg global – rj – mar/2003 a nov/2003\n- implementação do peoplesoft erp e crm e construção da integração entre as duas soluções na aga linde healthcare\n- análise de gaps\n- definição das tabelas depara para a interface\n- desenvolvimento de especificações técnicas\n- desenvolvimento de gaps no erp e crm\n- desenvolvimento dos programas de interface com sqr para o erp e o crm (cliente, contratos e field service)\n\n\natos origin/peoplefriend – rj – jun/2002 a fev/2003\n- suporte a produção na shell brasil:\n- suporte as demandas de produção\n- desenvolvimento e manutenção de sqr e peoplecode\n\n\nhqs consulting – rj/sp – dez/1999 a fev/2002\n- suporte ao peoplesoft hr de produção na michellin – rj. atuei em parceria técnica com a deloitte:\n- desenvolvimento de novos relatórios utilizando sqr e crystal report\n- desenvolvimento de novas querys com o query manager\n- importação de dados com o data mover\n\n- implementação da suíte financeira do peoplesoft 7.5 em parceria com a pricewaterhousecoopers na mbr - bh:\n- desenvolvimento de novos gaps para o ap, ar e tr.\n- desenvolvimento de novos relatórios com sqr e crystal report\n- desenvolvimento de interface para o ap e ar\n- coordenação de atividades técnicas\n\n- suporte ao peoplesoft hr 7.5 de produção em parceria com a hunter group na gm (general motors) - sp:\n- desenvolvimento de novos relatórios com sqr e crystal report\n- extração de dados para a equipe de sap\n\n- suporte técnico ao peoplesoft 7.5 de produção na sul américa seguros - rj:\n- desenvolvimento de novos relatórios com nvision e query manager\n- revisão e reconstrução da árvore de departamentos\n\n- suporte técnico ao peoplesoft 7.5 de produção na companhia de seguros aliança do brasil - rj:\n- desenvolvimento de gaps para: ap, ar, po, gl e hr\n- desenvolvimento de novas interfaces para: ap, ar, po, gl e hr\n",

          "nivel_profissional_vaga": "Sênior",

          "nivel_ingles_vaga": "Nenhum",

          "nivel_ingles": "Avançado",

          "nivel_academico": "Ensino Superior Completo"
          }


```

Exemplo de retorno: 

```bash
          {"match_score":0.04189454986361654,
           "prediction":1,
           "prob_contratado":0.65
          }
```
## 🐳 Deploy com Docker

### 1. Build da imagem

```bash
docker build -t fiap-ai-recruiter .
```
### 2. Run da API via Docker

```bash
docker run -it --rm -p 5000:5000 -p 5001:5001 fiap-ai-recruiter
```



