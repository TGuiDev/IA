const tf = require('@tensorflow/tfjs');
const readlineSync = require('readline-sync');

// Definir um modelo simples de rede neural
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1], activation: 'linear' }));
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

// Função para treinar o modelo
async function treinarModelo() {
  const entradas = [];
  const saidas = [];

  console.log("Digite os dados de entrada e saída para ensinar o modelo.");
  let continuar = true;

  while (continuar) {
    const entrada = readlineSync.questionFloat("Digite um valor de entrada: ");
    const saida = readlineSync.questionFloat("Digite a saída correspondente para esse valor de entrada: ");

    entradas.push([entrada]);
    saidas.push([saida]);

    continuar = readlineSync.keyInYNStrict("Deseja adicionar mais um exemplo?");

    // Se já tiver exemplos suficientes (por exemplo, 5), comece o treinamento
    if (entradas.length >= 5) {
      console.log("Começando o treinamento...");
      const treinoEntradas = tf.tensor2d(entradas);
      const treinoSaidas = tf.tensor2d(saidas);

      await model.fit(treinoEntradas, treinoSaidas, {
        epochs: 100,
        verbose: 0,
      });

      console.log("Treinamento concluído!");
      console.log("Modelo pronto para fazer previsões.");
      break;
    }
  }

  // Testar o modelo com novos dados
  const novoValor = readlineSync.questionFloat("Digite um novo valor para prever: ");
  const previsao = model.predict(tf.tensor2d([[novoValor]])).arraySync();
  console.log(`A previsão para ${novoValor} é: ${previsao[0][0]}`);
}

// Iniciar o processo de treinamento
treinarModelo().then(() => {
  console.log("Treinamento e previsão concluídos.");
}).catch(err => {
  console.error("Erro durante o processo: ", err);
});



