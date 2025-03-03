use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use stop_words::LANGUAGE;

const THRESHOLD: f64 = 0.5;

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
            .into_iter()
            .zip(encoding.get_ids().into_iter())
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

pub struct EmailClassifier {
    pub p_spam: f64,
    pub p_word: HashMap<u32, f64>,
    pub p_word_given_spam: HashMap<u32, f64>,
    pub length: f64,
    pub tokenizer: Tokenizer,
    pub s: f64,
}

impl From<Vec<Email>> for EmailClassifier {
    fn from(emails: Vec<Email>) -> Self {
        let length = emails.len();

        let p_spam = emails.iter().filter(|email| email.is_spam()).count();

        let mut p_word = HashMap::new();
        let mut p_word_given_spam = HashMap::new();

        let tokenizer = Tokenizer::default();

        for email in &emails {
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
}
