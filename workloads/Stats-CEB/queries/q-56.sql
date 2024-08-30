SELECT COUNT(*)
FROM tags AS t,
  posts AS p,
  users AS u,
  votes AS v,
  badges AS b
WHERE p.Id = t.ExcerptPostId
  AND u.Id = v.UserId
  AND u.Id = b.UserId
  AND u.Id = p.OwnerUserId
  AND u.DownVotes >= 0;
