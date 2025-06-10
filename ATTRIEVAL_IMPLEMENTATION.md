# ATTRIEVAL Implementation Guide

**Attention-guided Retrieval for Long-Context Reasoning**

Implementation of the ATTRIEVAL method from: *"Attention Reveals More Than Tokens: Training-Free Long-Context Reasoning with Attention-guided Retrieval"* by Zhang et al.

## 📖 Overview

ATTRIEVAL is a training-free algorithm that leverages attention weights from Chain-of-Thought (CoT) tokens to retrieve relevant facts from long contexts and improve reasoning performance. It addresses the key limitation that models often struggle to retrieve implicit facts during multi-hop reasoning tasks.

## 🔬 The Problem

In long-context reasoning tasks:
- Models excel at explicit fact retrieval but struggle with implicit relationships
- Chain-of-Thought prompting helps but doesn't fully resolve attention dispersion
- Performance degrades as context length increases, even when all necessary facts are present
- The primary bottleneck is **poor recall of implicit facts**, not faulty reasoning

## 🎯 The Solution: ATTRIEVAL Algorithm

ATTRIEVAL bridges the gap between retrieval and reasoning by:
1. **Multi-layer attention aggregation** - Uses attention weights from the last 25% of model layers
2. **Fact segmentation** - Breaks context into discrete, scoreable facts  
3. **Attention sink filtering** - Removes frequently attended but uninformative tokens
4. **Cross-evaluation** - Identifies "retriever" vs "reasoner" tokens for better scoring
5. **Fact scoring and selection** - Ranks facts by aggregated attention scores

## 📊 Step-by-Step Implementation

### Step 1: Multi-Layer Attention Aggregation

**Equation 4 from the paper:**
```
Ā_{t,i} = (1/|L|) * (1/H) * Σ_{l∈L} Σ_{h=1}^H A^{(l,h)}_{t,i}
```

- **L**: Last 25% of transformer layers (they better ground to context)
- **H**: Number of attention heads per layer
- **A^{(l,h)}_{t,i}**: Attention weight from generated token t to input token i in layer l, head h

```python
def _aggregate_attention_weights(self, attention_data):
    """Aggregate attention across multiple layers and heads."""
    attention_weights = attention_data['attention_weights']
    num_layers = len(attention_weights)
    
    # Select last 1/4 layers
    start_layer = int(num_layers * (1 - self.config.layer_fraction))
    selected_layers = list(range(start_layer, num_layers))
    
    # Aggregate across selected layers and heads
    aggregated = None
    for layer_idx in selected_layers:
        layer_attention = attention_weights[layer_idx]  # (num_heads, seq_len, seq_len)
        layer_avg = np.mean(layer_attention, axis=0)    # Average across heads
        
        if aggregated is None:
            aggregated = layer_avg
        else:
            aggregated += layer_avg
    
    return aggregated / len(selected_layers)  # Average across layers
```

### Step 2: Context Segmentation into Facts

Break the input context into discrete facts based on punctuation boundaries:

```python
def _segment_context_into_facts(self, context):
    """Segment input context into discrete facts."""
    sentences = re.split(r'[.!?]+', context)
    
    facts = []
    token_offset = 0
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence.split()) >= self.config.min_fact_length:
            fact_tokens = self.extractor.tokenizer.encode(sentence)
            
            fact_info = {
                'id': i,
                'text': sentence,
                'token_indices': list(range(token_offset, token_offset + len(fact_tokens))),
                'length': len(fact_tokens)
            }
            facts.append(fact_info)
            token_offset += len(fact_tokens)
    
    return facts
```

### Step 3: Top-k Token Identification

**Equation 5 from the paper:**
```
T_t = arg top-k(Ā_{t,i})
```

For each generated CoT token, identify the top-k input tokens with highest attention:

```python
def _identify_top_k_tokens(self, aggregated_attention):
    """Find top-k attended tokens for each generated token."""
    cot_tokens, context_tokens = aggregated_attention.shape
    top_k_tokens = {}
    
    for t in range(cot_tokens):
        attention_scores = aggregated_attention[t, :]
        top_k_indices = np.argsort(attention_scores)[-self.config.top_k:]
        top_k_tokens[t] = top_k_indices.tolist()
    
    return top_k_tokens
```

### Step 4: Attention Sink Filtering

**Equation 6 from the paper:**
```
f(c) = (1/T) * Σ_{t=1}^T I{c ∈ {c(i): i ∈ T_t}}
```

Filter out facts that appear too frequently in top-k positions (attention sinks):

```python
def _filter_attention_sinks(self, facts, top_k_tokens, all_tokens):
    """Filter out attention sink facts."""
    fact_frequencies = defaultdict(int)
    total_cot_tokens = len(top_k_tokens)
    
    # Map tokens to facts
    token_to_fact = {}
    for fact in facts:
        for token_idx in fact['token_indices']:
            if token_idx < len(all_tokens):
                token_to_fact[token_idx] = fact['id']
    
    # Count frequencies
    for t, top_tokens in top_k_tokens.items():
        facts_in_top_k = set()
        for token_idx in top_tokens:
            if token_idx in token_to_fact:
                facts_in_top_k.add(token_to_fact[token_idx])
        
        for fact_id in facts_in_top_k:
            fact_frequencies[fact_id] += 1
    
    # Filter based on threshold
    filtered_facts = []
    for fact in facts:
        frequency = fact_frequencies[fact['id']] / total_cot_tokens
        if frequency < self.config.frequency_threshold:
            fact['frequency'] = frequency
            filtered_facts.append(fact)
    
    return filtered_facts
```

### Step 5: Fact Scoring

**Equation 7 from the paper:**
```
s(c) = (1/|I_c|) * (1/T) * Σ_{i∈I_c} Σ_{t=1}^T Ā_{t,i}
```

Score each fact by its average attention across all CoT tokens:

```python
def _score_facts(self, facts, attention_weights, all_tokens):
    """Score facts based on attention weights."""
    fact_scores = {}
    num_cot_tokens, num_context_tokens = attention_weights.shape
    
    for fact in facts:
        fact_id = fact['id']
        token_indices = [idx for idx in fact['token_indices'] if idx < num_context_tokens]
        
        if not token_indices:
            fact_scores[fact_id] = 0.0
            continue
        
        # Calculate average attention across all CoT tokens and fact tokens
        total_attention = 0.0
        for token_idx in token_indices:
            for cot_token in range(num_cot_tokens):
                total_attention += attention_weights[cot_token, token_idx]
        
        # Average by number of tokens in fact and number of CoT tokens
        avg_attention = total_attention / (len(token_indices) * num_cot_tokens)
        fact_scores[fact_id] = float(avg_attention)
    
    return fact_scores
```

## 🚀 Usage Examples

### Basic Usage

```python
from attention_viz import AttentionExtractor, AttrievelRetriever, AttrievelConfig, load_model_and_tokenizer

# Load model
model, tokenizer = load_model_and_tokenizer("gpt2")

# Initialize ATTRIEVAL
extractor = AttentionExtractor(model, tokenizer)
config = AttrievelConfig(
    layer_fraction=0.25,      # Use last 25% of layers
    top_k=50,                 # Top 50 tokens per CoT token
    frequency_threshold=0.99, # Filter attention sinks
    max_facts=10             # Retrieve top 10 facts
)
retriever = AttrievelRetriever(extractor, config)

# Run ATTRIEVAL
result = retriever.retrieve_facts(
    context="Your long context with facts...",
    question="Your question...",
    cot_response="Chain-of-thought response...",
    use_cross_evaluation=True
)

# View results
print(retriever.visualize_retrieved_facts(result))
```

## 📊 Example Results

### Input Context
```
One of the special magic messages is: The price of oranges is 73 USD.
Many fruits are available in the market today.
One of the special magic messages is: The price of nectarines is 32 USD cheaper than the price of oranges.
The local grocery store has been in business for over 20 years.
One of the special magic messages is: The price of apples is 15 USD more expensive than the price of nectarines.
One of the special magic messages is: The price of bananas is half the price of apples.
```

### Question
"What is the price of bananas?"

### Retrieved Facts (with scores)
1. **[0.0061]** One of the special magic messages is: The price of nectarines is 32 USD cheaper than the price of oranges
2. **[0.0053]** Many fruits are available in the market today  
3. **[0.0049]** The weather is sunny and perfect for shopping
4. **[0.0009]** One of the special magic messages is: The price of apples is 15 USD more expensive than the price of nectarines

## 🎯 Key Benefits

1. **Training-free** - No model fine-tuning required
2. **Context-length robust** - Performance maintained on long contexts  
3. **Implicit fact retrieval** - Surfaces relationships not explicitly mentioned in CoT
4. **Attention sink filtering** - Reduces noise from frequently attended tokens
5. **Cross-evaluation** - Distinguishes retriever from reasoner tokens

## 📈 Performance Improvements

Based on the paper's results:
- **+47% accuracy** on Deduction benchmark (synthetic reasoning)
- **+11% accuracy** on MuSiQue dataset (real-world QA) 
- **Consistent improvements** across multiple model sizes and datasets
- **Maintained short-context performance** - no degradation on short contexts

## 📁 Generated Files

The implementation creates several output files:

- **`attrieval_results.json`**: Detailed results in JSON format
- **`attrieval_analysis_report.md`**: Human-readable analysis report  
- **`attrieval_attention_heatmap.png`**: Visualization of aggregated attention
- **`attrieval_fact_scores.png`**: Bar chart of retrieved fact scores
- **`attrieval_algorithm_flow.png`**: Algorithm flow diagram

## 🤝 Testing the Implementation

Run the simple test:
```bash
source attnviz/bin/activate
python examples/simple_attrieval_test.py
```

Run the full demonstration:
```bash
python examples/attrieval_demo.py
```

---

This implementation demonstrates how attention mechanisms in transformers can be leveraged to improve long-context reasoning without requiring additional training, providing a practical tool for enhancing model performance on complex reasoning tasks. 