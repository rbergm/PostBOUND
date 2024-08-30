SELECT COUNT(*)
FROM posts AS p, tags AS t, votes AS v
WHERE p.Id = t.ExcerptPostId
  AND p.OwnerUserId = v.UserId
  AND p.CreationDate >= CAST('2010-07-20 02:01:05' AS timestamp);
