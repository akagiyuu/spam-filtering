use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};
use stop_words::LANGUAGE;

const THRESHOLD: f64 = 0.5;

#[derive(Serialize, Deserialize)]
pub struct Tokenizer {
    tokenizer: tokenizers::Tokenizer,
    stopwords: HashSet<String>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Tokenizer {
            tokenizer: tokenizers::Tokenizer::from_pretrained("bert-base-uncased", None).unwrap(),
            stopwords: HashSet::from_iter(stop_words::get(LANGUAGE::English)),
        }
    }
}

impl Tokenizer {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text = text.to_lowercase();

        let encoding = self.tokenizer.encode(text.clone(), false).unwrap();

        encoding
            .get_tokens()
            .iter()
            .zip(encoding.get_ids())
            .filter_map(|(token, &id)| {
                if self.stopwords.contains(token) || token.chars().any(|c| !c.is_alphanumeric()) {
                    None
                } else {
                    Some(id)
                }
            })
            .collect()
    }
}

#[derive(Deserialize)]
pub struct Email {
    content: String,
    category: u8,
}

impl Email {
    pub fn is_spam(&self) -> bool {
        self.category == 1
    }
}

#[derive(Deserialize, Serialize)]
pub struct EmailClassifier {
    pub p_spam: f64,
    pub p_word: HashMap<u32, f64>,
    pub p_word_given_spam: HashMap<u32, f64>,
    pub length: f64,
    #[serde(skip)]
    pub tokenizer: Tokenizer,
    pub s: f64,
}

impl From<&[Email]> for EmailClassifier {
    fn from(emails: &[Email]) -> Self {
        let length = emails.len() as f64;

        let p_spam = emails.iter().filter(|email| email.is_spam()).count() as f64 / length;

        let mut p_word = HashMap::new();
        let mut p_word_given_spam = HashMap::new();

        let tokenizer = Tokenizer::default();

        for email in emails {
            for token in tokenizer.encode(&email.content) {
                *p_word.entry(token).or_default() += 1.;
                if email.is_spam() {
                    *p_word_given_spam.entry(token).or_default() += 1.;
                }
            }
        }

        Self {
            p_spam,
            p_word,
            p_word_given_spam,
            length,
            tokenizer,
            s: 1.,
        }
    }
}

impl EmailClassifier {
    pub fn load(path: impl AsRef<Path>) -> Self {
        let raw = fs::read(path).unwrap();
        bincode::deserialize(&raw).unwrap()
    }
    pub fn save(&self, path: impl AsRef<Path>) {
        let raw = bincode::serialize(self).unwrap();
        fs::write(path, raw).unwrap();
    }

    pub fn _prob(&self, token: u32) -> f64 {
        let n = self.p_word.get(&token).copied().unwrap_or(0.) * self.length;

        (self.s * self.p_spam + n * self.p_word.get(&token).copied().unwrap_or(0.)) / (self.s + n)
    }

    pub fn prob(&self, content: &str) -> f64 {
        let tokens = self.tokenizer.encode(content);

        let log_prob = tokens
            .into_iter()
            .map(|token| self._prob(token))
            .map(|prob| (1. - prob).ln() - prob.ln())
            .sum::<f64>();

        1. / (1. + log_prob.exp())
    }

    pub fn is_spam(&self, content: &str) -> bool {
        self.prob(content) > THRESHOLD
    }
}

fn main() {
    let data = fs::OpenOptions::new().read(true).open("data.csv").unwrap();
    let mut reader = csv::Reader::from_reader(data);

    let mut data = reader
        .deserialize::<Email>()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    data.shuffle(&mut rand::rng());

    let (train, test) = data.split_at(data.len() * 8 / 10);

    let classifier = EmailClassifier::from(train);

    classifier.save("model");

    let test_length = test.len();
    let correct_count = test
        .iter()
        .filter(|email| classifier.is_spam(&email.content) == email.is_spam())
        .count();

    println!("Accuracy: {}", correct_count as f64 / test_length as f64);
}
