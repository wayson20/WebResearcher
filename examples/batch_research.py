#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Research Example

Demonstrates how to process multiple questions in batch mode.
"""
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import sys
sys.path.append("..")

load_dotenv()

from webresearcher import WebResearcherAgent


async def batch_research(questions, output_dir="./results"):
    """
    Process multiple research questions in batch.
    
    Args:
        questions: List of question strings or dicts with 'question' and 'ground_truth'
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    agent = WebResearcherAgent(
        model="gpt-4o",
        function_list=["search", "python"]
    )
    
    # Process each question
    results = []
    for i, item in enumerate(questions, 1):
        # Parse item
        if isinstance(item, str):
            question = item
            ground_truth = None
        else:
            question = item.get('question', item)
            ground_truth = item.get('ground_truth') or item.get('answer')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {i}/{len(questions)}: {question[:100]}...")
        logger.info(f"{'='*80}")
        
        try:
            # Run research
            result = await agent.run(question)
            
            # Add metadata
            result['index'] = i
            result['ground_truth'] = ground_truth
            result['success'] = result.get('termination', '')
            results.append(result)
            
            # Save individual result
            output_file = output_path / f"result_{i:03d}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Print summary
            print(f"\n✅ Question {i} completed")
            print(f"   Question: {question}")
            print(f"   Answer: {result['prediction']}")
            if ground_truth:
                print(f"   Ground Truth: {ground_truth}")
            print(f"   Saved to: {output_file}")
        except Exception as e:
            logger.error(f"❌ Question {i} failed: {e}")
            results.append({
                'index': i,
                'question': question,
                'error': str(e),
                'success': False
            })
    
    # Save summary
    summary = {
        'total': len(questions),
        'results': results
    }
    
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("BATCH RESEARCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total Questions: {summary['total']}")
    print(f"\nResults saved to: {output_dir}/")
    
    return summary


async def main():
    """Example usage"""
    # Define questions to research
    questions = [
        {
            "question": "刘翔破纪录时候是多少岁?",
            "answer": "23岁差2天"
        },
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        "Who invented the World Wide Web?",
    ]
    
    # Run batch research
    summary = await batch_research(questions, output_dir="./outputs")
    
    return summary


if __name__ == "__main__":
    asyncio.run(main())

